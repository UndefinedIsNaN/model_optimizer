<#
.SYNOPSIS
    Запускает все три варианта прунинга для указанной модели
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Repo,
    
    [int]$RecoverySteps = 0,
    
    [string]$OutputDir = "./output",
    
    [string]$WorkDir = "./work",
    
    [string]$HfToken = $null,
    
    [int]$CalSamples = 256,
    
    [int]$MaxSeqLen = 512,
    
    [string]$VenvPath = "..\..\venv"
)

# Определяем пути вручную
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# Корень проекта: поднимаемся на 2 уровня ВВЕРХ от скрипта
# C:\...\model_optimizer\installation_scripts\pc
# → C:\...\model_optimizer\installation_scripts
# → C:\...\model_optimizer  ← корень проекта

$parent1 = Split-Path -Parent $scriptDir      # installation_scripts
$projectRoot = Split-Path -Parent $parent1       # model_optimizer

# Путь к Prune.py
$pruneScript = Join-Path $projectRoot "Prune.py"

# Путь к venv
if ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $venvFullPath = $VenvPath
} else {
    $venvFullPath = Join-Path $projectRoot $VenvPath
}

# Нормализуем путь (убираем ..\)
$venvFullPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $VenvPath))
$venvPython = Join-Path $venvFullPath "Scripts\python.exe"

# Абсолютные пути для output/work
$OutputDir = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDir))
$WorkDir = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $WorkDir))

# Отладка
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "  ОТЛАДКА ПУТЕЙ"
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "Script dir:    $scriptDir"
Write-Host "Parent1:       $parent1"
Write-Host "Project root:  $projectRoot"
Write-Host "Venv path:     $venvFullPath"
Write-Host "Python exe:    $venvPython"
Write-Host "Prune.py:      $pruneScript"
Write-Host ""

# Проверка существования venv
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Python не найден: $venvPython" -ForegroundColor Red
    
    # Ищем venv в корне проекта автоматически
    $altVenv = Join-Path $projectRoot "venv"
    $altPython = Join-Path $altVenv "Scripts\python.exe"
    
    if (Test-Path $altPython) {
        Write-Host "[INFO] Найден альтернативный venv: $altPython" -ForegroundColor Green
        $venvPython = $altPython
        $venvFullPath = $altVenv
    } else {
        Write-Host "[ERROR] Папка venv не существует: $venvFullPath" -ForegroundColor Red
        Write-Host "[INFO] Искали также: $altPython" -ForegroundColor Yellow
        
        # Показываем содержимое корня проекта
        Write-Host "[INFO] Содержимое $projectRoot :" -ForegroundColor Yellow
        Get-ChildItem $projectRoot -Directory | ForEach-Object {
            Write-Host "  📁 $($_.Name)" -ForegroundColor Gray
        }
        exit 1
    }
}

Write-Host "[OK] Python найден: $venvPython" -ForegroundColor Green

# Проверка Prune.py
if (-not (Test-Path $pruneScript)) {
    Write-Host "[ERROR] Prune.py не найден: $pruneScript" -ForegroundColor Red
    exit 1
}

# Основной вывод
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "  ЗАПУСК ВСЕХ ВАРИАНТОВ ПРУНИНГА"
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "[INFO] Модель: $Repo"
Write-Host "[INFO] Recovery: $RecoverySteps шагов"
Write-Host "[INFO] Output: $OutputDir"
Write-Host "[INFO] Work: $WorkDir"

# Создаём папки
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

# Варианты прунинга
$variants = @("magnitude_20", "magnitude_30", "structured")
$failed = @()

for ($i = 0; $i -lt $variants.Count; $i++) {
    $variant = $variants[$i]
    $num = $i + 1
    
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "  [$num/$($variants.Count)] Прунинг: $variant"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Формируем команду
# Формируем команду
$arguments = @(
    "`"$pruneScript`"",
    "--repo", "`"$Repo`"",
    "--pruning", $variant,
    "--output-dir", "`"$OutputDir`"",
    "--work-dir", "`"$WorkDir`"",
    "--cal-samples", $CalSamples,
    "--max-seq-len", $MaxSeqLen
)

if ($RecoverySteps -gt 0) {
    $arguments += "--recovery-steps"
    $arguments += $RecoverySteps
}

# Квантизация только если явно указана
if ($QuantType) {
    $arguments += "--quant-type"
    $arguments += $QuantType
}

if ($HfToken) {
    $arguments += "--hf-token"
    $arguments += "`"$HfToken`""
}
    
    Write-Host "[INFO] Запуск: python $([string]::Join(" ", $arguments))"
    Write-Host ""
    
    # Запускаем процесс
    $process = Start-Process -FilePath $venvPython -ArgumentList $arguments -Wait -NoNewWindow -PassThru
    $exitCode = $process.ExitCode
    
    if ($exitCode -ne 0) {
        Write-Host "[ERROR] Ошибка при $variant (код $exitCode)" -ForegroundColor Red
        $failed += $variant
    } else {
        Write-Host "[OK] Завершено: $variant" -ForegroundColor Green
    }
    
    if ($i -lt $variants.Count - 1) {
        Write-Host "[WARN] Пауза 3 сек..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }
}

# Итоги
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "  РЕЗУЛЬТАТЫ"
Write-Host "═══════════════════════════════════════════════════════════"

if ($failed.Count -eq 0) {
    Write-Host "[OK] Все варианты выполнены!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Ошибки в: $($failed -join ', ')" -ForegroundColor Red
}

Write-Host "[INFO] Результаты: $OutputDir"