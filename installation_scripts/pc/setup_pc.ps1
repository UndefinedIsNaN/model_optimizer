<#
Установка на ПК (обработка моделей)

Использование:
  .\scripts\pc\install.ps1
#>

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectDir = (Resolve-Path "$ScriptDir\..\..").Path
$VenvDir = "$ProjectDir\venv"

Write-Host ""
Write-Host "Установка на ПК (обработка моделей)" -ForegroundColor Cyan
Write-Host "Проект: $ProjectDir" -ForegroundColor Gray
Write-Host ""

# 1. Системные зависимости
Write-Host "[1/6] Проверка системных зависимостей" -ForegroundColor Green

$Missing = @()

# Проверка Python
try {
    $PyVersion = python --version 2>&1
    Write-Host "  Найден: $PyVersion" -ForegroundColor Gray
} catch {
    $Missing += "python"
}

# Проверка Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $Missing += "git"
} else {
    Write-Host "  Найден: git" -ForegroundColor Gray
}

# Проверка CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $Missing += "cmake"
} else {
    Write-Host "  Найден: cmake" -ForegroundColor Gray
}

# Проверка компилятора (cl.exe для MSVC или gcc)
$HasCompiler = $false
if (Get-Command cl -ErrorAction SilentlyContinue) {
    Write-Host "  Найден: MSVC (cl.exe)" -ForegroundColor Gray
    $HasCompiler = $true
} elseif (Get-Command gcc -ErrorAction SilentlyContinue) {
    Write-Host "  Найден: GCC" -ForegroundColor Gray
    $HasCompiler = $true
}

if ($Missing.Count -gt 0 -or -not $HasCompiler) {
    Write-Host "ОШИБКА: не найдены: $($Missing -join ', ')" -ForegroundColor Red
    if (-not $HasCompiler) {
        Write-Host "  Не найден компилятор C++ (MSVC или GCC)" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Установите:" -ForegroundColor Yellow
    Write-Host "  1. Python 3.10+ с python.org (отметьте 'Add to PATH')" -ForegroundColor White
    Write-Host "  2. Git с git-scm.com" -ForegroundColor White
    Write-Host "  3. Visual Studio Build Tools или MSVC" -ForegroundColor White
    Write-Host "     - https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor White
    Write-Host "     - Или: winget install Microsoft.VisualStudio.2022.BuildTools" -ForegroundColor White
    Write-Host "  4. CMake: winget install Kitware.CMake" -ForegroundColor White
}
Write-Host "  OK" -ForegroundColor Green

# 2. Python venv
Write-Host "[2/6] Python виртуальное окружение" -ForegroundColor Green

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
    Write-Host "  venv создан" -ForegroundColor Gray
} else {
    Write-Host "  venv уже существует" -ForegroundColor Gray
}

& "$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel -q
$PyVer = python --version
Write-Host "  Python: $PyVer" -ForegroundColor Gray

# 3. PyTorch
Write-Host "[3/6] PyTorch" -ForegroundColor Green

# Проверка CUDA через nvidia-smi
$HasCuda = $false
try {
    $null = nvidia-smi 2>$null
    $HasCuda = $true
} catch {}

# Проверка уже установленного PyTorch с CUDA
$TorchHasCuda = $false
try {
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaAvailable -eq "True") {
        $TorchHasCuda = $true
    }
} catch {}

if ($TorchHasCuda) {
    Write-Host "  CUDA обнаружена, PyTorch уже установлен" -ForegroundColor Gray
} elseif ($HasCuda) {
    Write-Host "  NVIDIA GPU обнаружена, ставим PyTorch с CUDA" -ForegroundColor Gray
    pip install torch -q 2>$null
    if ($LASTEXITCODE -ne 0) {
        pip install torch -q
    }
} else {
    Write-Host "  GPU не обнаружена, ставим CPU-версию" -ForegroundColor Gray
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>$null
    if ($LASTEXITCODE -ne 0) {
        pip install torch -q
    }
}

# 4. Зависимости
Write-Host "[4/6] Python-зависимости" -ForegroundColor Green

pip install -q transformers>=4.36 huggingface_hub>=0.20 datasets>=2.16 safetensors sentencepiece protobuf psutil numpy

Write-Host "  OK" -ForegroundColor Green

# 5. Директории
Write-Host "[5/6] Директории" -ForegroundColor Green

$Dirs = @("work", "output", "bench_results", "logs")
foreach ($d in $Dirs) {
    $path = "$ProjectDir\$d"
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

Write-Host "  work/          -- промежуточные файлы" -ForegroundColor Gray
Write-Host "  output/        -- готовые .gguf" -ForegroundColor Gray
Write-Host "  bench_results/ -- результаты бенчмарков" -ForegroundColor Gray
Write-Host "  logs/          -- логи" -ForegroundColor Gray

# 6. Проверка
Write-Host "[6/6] Проверка" -ForegroundColor Green

$Errors = 0

function Test-Module {
    param([string]$Name, [string]$ImportCmd)
    
    try {
        $null = Invoke-Expression $ImportCmd 2>&1
        Write-Host "  OK  $Name" -ForegroundColor Green
    } catch {
        Write-Host "  ERR $Name" -ForegroundColor Red
        $script:Errors++
    }
}

Test-Module "Python"       "python --version"
Test-Module "PyTorch"      "python -c 'import torch; print(torch.__version__)'"
Test-Module "Transformers" "python -c 'import transformers'"
Test-Module "HF Hub"       "python -c 'import huggingface_hub'"
Test-Module "datasets"     "python -c 'import datasets'"
Test-Module "cmake"        "cmake --version"
Test-Module "git"          "git --version"

# Проверка GPU
try {
    $cudaCheck = python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" 2>$null
    if ($cudaCheck) {
        Write-Host "  OK  CUDA GPU: $cudaCheck" -ForegroundColor Green
    }
} catch {
    try {
        $mpsCheck = python -c "import torch; assert torch.backends.mps.is_available()" 2>$null
        Write-Host "  OK  Apple MPS" -ForegroundColor Green
    } catch {
        Write-Host "  --  Нет GPU (будет CPU)" -ForegroundColor Gray
    }
}

Write-Host ""
if ($Errors -gt 0) {
    Write-Host "ОШИБКА: $Errors проблем" -ForegroundColor Red
    exit 1
}

Write-Host "ПК готов." -ForegroundColor Green
Write-Host ""
Write-Host "Активация:" -ForegroundColor Yellow
Write-Host "  & $VenvDir\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Один пайплайн:" -ForegroundColor Yellow
Write-Host "  .\scripts\pc\run_pipeline.ps1 Qwen/Qwen2-0.5B structured ptq Q4_K_M prune_first" -ForegroundColor White
Write-Host ""
Write-Host "Все 9 комбинаций:" -ForegroundColor Yellow
Write-Host "  .\scripts\pc\run_all_combinations.ps1 Qwen/Qwen2-0.5B" -ForegroundColor White
Write-Host ""
Write-Host "Отправить на одноплатники:" -ForegroundColor Yellow
Write-Host "  .\scripts\pc\deploy.ps1" -ForegroundColor White
Write-Host ""

# Пауза если запущено двойным кликом
if ($Host.Name -eq 'ConsoleHost' -and -not $env:TERM) {
    Write-Host "Нажмите Enter для закрытия..." -ForegroundColor Cyan
    Read-Host
}