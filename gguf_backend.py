"""
Обёртка над llama.cpp: клонирование, сборка, конвертация HF->GGUF, квантизация.
Поддержка Windows, Linux, macOS.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"


class GGUFBackend:

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.llama_dir = work_dir / "llama.cpp"
        self.build_dir = self.llama_dir / "build"
        self._cloned = False
        self._built = False

    #  Публичный API

    def convert_hf_to_gguf(
        self, model_dir: Path, output: Path, dtype: str = "f16",
    ) -> Path:
        """HuggingFace checkpoint -> GGUF. Нужен только Python, cmake не нужен."""
        self._ensure_cloned()
        self._install_convert_deps()

        script = self.llama_dir / "convert_hf_to_gguf.py"
        if not script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py не найден в {self.llama_dir}\n"
                f"Возможно, структура llama.cpp изменилась."
            )

        logger.info("HF -> GGUF (%s): %s", dtype, output.name)
        self._run([
            sys.executable, str(script),
            str(model_dir),
            "--outfile", str(output),
            "--outtype", dtype,
        ])
        return output

    def quantize(self, src: Path, dst: Path, quant_type: str) -> Path:
        """Квантизация GGUF -> GGUF. Нужен собранный llama-quantize."""
        self._ensure_built()

        qbin = self._find_quantize_bin()
        if qbin is None:
            raise RuntimeError(
                "llama-quantize не найден после сборки.\n"
                + self._build_help_message()
            )

        logger.info("Quantize %s -> %s (%s)", src.name, dst.name, quant_type)
        self._run([str(qbin), str(src), str(dst), quant_type])
        return dst

    #  Клонирование

    def _ensure_cloned(self) -> None:
        if self._cloned:
            return
        if self.llama_dir.exists() and (self.llama_dir / ".git").exists():
            logger.info("llama.cpp уже склонирован: %s", self.llama_dir)
            self._cloned = True
            return

        self._check_command("git", install_hint=self._git_hint())

        logger.info("Клонируем llama.cpp ...")
        self._run([
            "git", "clone", "--depth=1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(self.llama_dir),
        ])
        self._cloned = True

    #  Зависимости для convert_hf_to_gguf.py

    def _install_convert_deps(self) -> None:
        """Установить Python-зависимости скрипта конвертации."""
        for name in (
            "requirements.txt",
            "requirements/requirements-convert_hf_to_gguf.txt",
        ):
            req = self.llama_dir / name
            if req.exists():
                self._run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                    check=False,
                )

    #  Сборка llama-quantize

    def _ensure_built(self) -> None:
        if self._built:
            return

        self._ensure_cloned()

        # Может уже собран
        if self._find_quantize_bin() is not None:
            logger.info("llama-quantize уже собран")
            self._built = True
            return

        # Проверяем cmake
        cmake_path = self._find_cmake()
        if cmake_path is None:
            raise RuntimeError(
                "cmake не найден.\n" + self._cmake_hint()
            )

        # Проверяем компилятор
        self._check_compiler()

        logger.info("Собираем llama.cpp ...")
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Шаг 1: cmake configure
        configure_cmd = [cmake_path, ".."]

        if IS_WINDOWS:
            # На Windows пробуем найти Visual Studio
            # cmake сам найдёт MSVC если установлен
            pass
        else:
            # Linux/macOS: проверяем CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    configure_cmd.append("-DGGML_CUDA=ON")
            except ImportError:
                pass

        logger.info("  cmake configure: %s", " ".join(configure_cmd))
        self._run(configure_cmd, cwd=self.build_dir)

        # Шаг 2: cmake build
        n_jobs = str(os.cpu_count() or 4)
        build_cmd = [
            cmake_path, "--build", ".", "--config", "Release",
            "-j", n_jobs,
        ]

        logger.info("  cmake build (%s потоков) ...", n_jobs)
        self._run(build_cmd, cwd=self.build_dir)

        # Проверяем результат
        qbin = self._find_quantize_bin()
        if qbin is None:
            raise RuntimeError(
                "Сборка завершилась, но llama-quantize не найден.\n"
                + self._build_help_message()
            )

        logger.info("  llama-quantize: %s", qbin)
        self._built = True

    # --------------------------------------------------
    #  Поиск бинарников
    # --------------------------------------------------

    def _find_quantize_bin(self) -> Optional[Path]:
        """Ищет llama-quantize в разных местах сборки."""
        candidates = [
            # Linux / macOS
            self.build_dir / "bin" / "llama-quantize",
            self.llama_dir / "build" / "bin" / "llama-quantize",
            # Windows (cmake --build с MSVC)
            self.build_dir / "bin" / "Release" / "llama-quantize.exe",
            self.build_dir / "bin" / "llama-quantize.exe",
            self.build_dir / "Release" / "llama-quantize.exe",
            self.build_dir / "bin" / "Debug" / "llama-quantize.exe",
            # Windows (MinGW / MSYS2)
            self.build_dir / "bin" / "llama-quantize.exe",
            # Старые версии llama.cpp
            self.build_dir / "quantize",
            self.build_dir / "quantize.exe",
        ]

        for c in candidates:
            if c.exists():
                return c

        # Рекурсивный поиск
        if self.build_dir.exists():
            pattern = "llama-quantize.exe" if IS_WINDOWS else "llama-quantize"
            found = list(self.build_dir.rglob(pattern))
            if found:
                return found[0]

        return None

    def _find_cmake(self) -> Optional[str]:
        """Ищет cmake в PATH и стандартных местах."""
        # Сначала в PATH
        cmake_in_path = shutil.which("cmake")
        if cmake_in_path:
            return cmake_in_path

        if IS_WINDOWS:
            # Стандартные места установки на Windows
            search_dirs = [
                Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "CMake" / "bin",
                Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "CMake" / "bin",
                Path(os.environ.get("LOCALAPPDATA", "")) / "CMake" / "bin",
                # chocolatey
                Path("C:\\ProgramData\\chocolatey\\bin"),
                # scoop
                Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims",
            ]

            for d in search_dirs:
                cmake_exe = d / "cmake.exe"
                if cmake_exe.exists():
                    logger.info("  cmake найден: %s", cmake_exe)
                    return str(cmake_exe)

        return None

    #  Проверка компилятора

    def _check_compiler(self) -> None:
        """Проверяет наличие C++ компилятора."""
        if IS_WINDOWS:
            # Проверяем cl.exe (MSVC)
            cl = shutil.which("cl")
            if cl:
                return

            # Проверяем g++ (MinGW)
            gpp = shutil.which("g++")
            if gpp:
                return

            # Пробуем найти MSVC через vswhere
            vswhere = Path(
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
            ) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

            if vswhere.exists():
                try:
                    result = subprocess.run(
                        [str(vswhere), "-latest", "-property", "installationPath"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        logger.info("  Visual Studio: %s", result.stdout.strip())
                        return
                except Exception:
                    pass

            raise RuntimeError(
                "C++ компилятор не найден.\n\n"
                "Установите один из:\n"
                "  1. Visual Studio Build Tools (рекомендуется):\n"
                "     https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
                "     При установке выберите 'C++ build tools'\n\n"
                "  2. MinGW-w64:\n"
                "     https://www.mingw-w64.org/\n\n"
                "После установки перезапустите терминал."
            )
        else:
            # Linux / macOS
            if shutil.which("g++") or shutil.which("c++") or shutil.which("clang++"):
                return

            raise RuntimeError(
                "C++ компилятор не найден.\n\n"
                "Установите:\n"
                "  Ubuntu/Debian: sudo apt install build-essential\n"
                "  Fedora:        sudo dnf install gcc-c++\n"
                "  macOS:         xcode-select --install\n"
            )

    #  Проверка команды

    def _check_command(self, name: str, install_hint: str = "") -> None:
        if shutil.which(name) is not None:
            return
        msg = f"Команда '{name}' не найдена."
        if install_hint:
            msg += "\n" + install_hint
        raise RuntimeError(msg)

    #  Запуск процессов

    @staticmethod
    def _run(
        cmd: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Запустить команду с логированием ошибок."""
        try:
            return subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=check,
                # На Windows нужен shell=False (по умолчанию)
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Не удалось запустить: {cmd[0]}\n"
                f"Убедитесь, что программа установлена и доступна в PATH.\n"
                f"Ошибка: {e}"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Команда завершилась с ошибкой (код {e.returncode}):\n"
                f"  {' '.join(cmd)}\n"
            ) from e

    #  Сообщения-подсказки

    @staticmethod
    def _cmake_hint() -> str:
        if IS_WINDOWS:
            return (
                "Установите cmake одним из способов:\n"
                "  1. Скачайте: https://cmake.org/download/\n"
                "     (при установке отметьте 'Add CMake to PATH')\n"
                "  2. Через chocolatey: choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System'\n"
                "  3. Через scoop: scoop install cmake\n"
                "  4. Через winget: winget install Kitware.CMake\n"
                "\nПосле установки перезапустите терминал."
            )
        else:
            return (
                "Установите cmake:\n"
                "  Ubuntu/Debian: sudo apt install cmake\n"
                "  Fedora:        sudo dnf install cmake\n"
                "  macOS:         brew install cmake\n"
            )

    @staticmethod
    def _git_hint() -> str:
        if IS_WINDOWS:
            return (
                "Установите git:\n"
                "  https://git-scm.com/download/win\n"
                "  Или: winget install Git.Git\n"
            )
        else:
            return (
                "Установите git:\n"
                "  Ubuntu/Debian: sudo apt install git\n"
                "  Fedora:        sudo dnf install git\n"
                "  macOS:         brew install git\n"
            )

    def _build_help_message(self) -> str:
        msg = "llama-quantize не удалось собрать.\n\n"

        if IS_WINDOWS:
            msg += (
                "На Windows сборка требует:\n"
                "  1. cmake (https://cmake.org/download/)\n"
                "  2. Visual Studio Build Tools или MinGW\n"
                "     https://visualstudio.microsoft.com/visual-cpp-build-tools/\n\n"
                "Альтернатива -- собрать вручную:\n"
                f"  cd {self.llama_dir}\n"
                "  mkdir build && cd build\n"
                "  cmake ..\n"
                "  cmake --build . --config Release\n\n"
                "Или скачайте готовый llama-quantize:\n"
                "  https://github.com/ggerganov/llama.cpp/releases\n"
                f"  и положите в {self.build_dir / 'bin'}\n"
            )
        else:
            msg += (
                "Убедитесь, что установлены:\n"
                "  sudo apt install build-essential cmake\n\n"
                "Попробуйте собрать вручную:\n"
                f"  cd {self.llama_dir}\n"
                "  mkdir build && cd build\n"
                "  cmake ..\n"
                "  cmake --build . --config Release -j$(nproc)\n"
            )

        return msg