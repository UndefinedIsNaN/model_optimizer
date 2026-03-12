#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

echo ""
echo "Установка на ПК (обработка моделей)"
echo "Проект: ${PROJECT_DIR}"
echo ""

# 1. Системные зависимости
echo "[1/6] Проверка системных зависимостей"

MISSING=()
command -v python3 >/dev/null 2>&1 || MISSING+=("python3")
command -v git     >/dev/null 2>&1 || MISSING+=("git")
command -v cmake   >/dev/null 2>&1 || MISSING+=("cmake")
command -v g++     >/dev/null 2>&1 || MISSING+=("g++ (build-essential)")

if (( ${#MISSING[@]} > 0 )); then
    echo "ОШИБКА: не найдены: ${MISSING[*]}"
    echo ""
    if command -v apt-get >/dev/null 2>&1; then
        echo "Установите: sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip"
    elif command -v brew >/dev/null 2>&1; then
        echo "Установите: brew install cmake python3 git"
    elif command -v pacman >/dev/null 2>&1; then
        echo "Установите: sudo pacman -S base-devel cmake git python"
    fi
    exit 1
fi
echo "  OK"

# 2. Python venv
echo "[2/6] Python виртуальное окружение"

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    echo "  venv создан"
else
    echo "  venv уже существует"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q
echo "  Python: $(python --version)"

# 3. PyTorch
echo "[3/6] PyTorch"

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo "  CUDA обнаружена, PyTorch уже установлен"
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "  NVIDIA GPU обнаружена, ставим PyTorch с CUDA"
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q \
        2>/dev/null || pip install torch -q
else
    echo "  GPU не обнаружена, ставим CPU-версию"
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q \
        2>/dev/null || pip install torch -q
fi

# 4. Зависимости
echo "[4/6] Python-зависимости"

pip install -q \
    transformers>=4.36 \
    huggingface_hub>=0.20 \
    datasets>=2.16 \
    safetensors \
    sentencepiece \
    protobuf \
    psutil \
    numpy

echo "  OK"

# 5. Директории
echo "[5/6] Директории"

mkdir -p "${PROJECT_DIR}"/{work,output,bench_results,logs}

echo "  work/          -- промежуточные файлы"
echo "  output/        -- готовые .gguf"
echo "  bench_results/ -- результаты бенчмарков"
echo "  logs/          -- логи"

# 6. Проверка
echo "[6/6] Проверка"

ERRORS=0

check() {
    local name="$1" cmd="$2"
    if eval "${cmd}" >/dev/null 2>&1; then
        echo "  OK  ${name}"
    else
        echo "  ERR ${name}"
        ERRORS=$((ERRORS + 1))
    fi
}

check "Python"       "python --version"
check "PyTorch"      "python -c 'import torch; print(torch.__version__)'"
check "Transformers" "python -c 'import transformers'"
check "HF Hub"       "python -c 'import huggingface_hub'"
check "datasets"     "python -c 'import datasets'"
check "cmake"        "cmake --version"
check "git"          "git --version"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "  OK  CUDA GPU: ${GPU_NAME}"
elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo "  OK  Apple MPS"
else
    echo "  --  Нет GPU (будет CPU)"
fi

echo ""
if (( ERRORS > 0 )); then
    echo "ОШИБКА: ${ERRORS} проблем"
    exit 1
fi

echo "ПК готов."
echo ""
echo "Активация:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Один пайплайн:"
echo "  bash scripts/pc/run_pipeline.sh Qwen/Qwen2-0.5B structured ptq Q4_K_M prune_first"
echo ""
echo "Все 9 комбинаций:"
echo "  bash scripts/pc/run_all_combinations.sh Qwen/Qwen2-0.5B"
echo ""
echo "Отправить на одноплатники:"
echo "  bash scripts/pc/deploy.sh"
echo ""