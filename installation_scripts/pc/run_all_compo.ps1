<#
Все 9 комбинаций для одной модели

Использование:
  .\scripts\pc\run_all_combinations.ps1 <repo> [quant_type] [доп. аргументы]

Примеры:
  .\scripts\pc\run_all_combinations.ps1 Qwen/Qwen2-0.5B
  .\scripts\pc\run_all_combinations.ps1 Qwen/Qwen2-0.5B Q5_K_M
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="Репозиторий модели")]
    [string]$Repo,

    [Parameter(Mandatory=$false)]
    [string]$QuantType = "Q4_K_M",

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# Фикс 1: Определение путей с проверкой существования
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$ProjectDir = Resolve-Path "$ScriptDir\..\.." | Select-Object -ExpandProperty Path

# Фикс 2: Проверка существования venv перед активацией
$VenvPath = "$ProjectDir\venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvPath)) {
    Write-Error "Виртуальное окружение не найдено: $VenvPath`nСначала запустите install.ps1"
    exit 1
}

# Фикс 3: Корректная активация venv (только для текущей сессии)
$env:PATH = "$ProjectDir\venv\Scripts;" + $env:PATH
& $VenvPath

Set-Location $ProjectDir

# Фикс 4: Создание директорий с проверкой
$LogDir = "$ProjectDir\logs"
$OutputDir = "$ProjectDir\output"
$WorkDir = "$ProjectDir\work"

@($LogDir, $OutputDir, $WorkDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Force -Path $_ | Out-Null
    }
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = "$LogDir\all_combos_$Timestamp.log"

# Фикс 5: Корректная работа Tee-Object (создаём файл если не существует)
if (-not (Test-Path $Log)) {
    New-Item -Path $Log -ItemType File -Force | Out-Null
}

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

Write-Log "Все 9 комбинаций"
Write-Log "  Repo:  $Repo"
Write-Log "  Quant: $QuantType"
Write-Log ""

$Total = 9
$Current = 0
$Failed = 0
$Succeeded = 0
$Times = @()

function Run-One {
    param(
        [string]$Pruning,
        [string]$Method,
        [string]$Order
    )

    $script:Current++
    Write-Log ""
    Write-Log "[$Current/$Total] $Pruning + $Method ($Order)"

    $StartTime = Get-Date

    # Фикс 6: Правильный формат аргументов для Python
    $PyArgs = @(
        "$ProjectDir\main.py",
        "--repo", $Repo,
        "--pruning", $Pruning,
        "--quant-method", $Method,
        "--quant-type", $QuantType,
        "--order", $Order,
        "--recovery-steps", "50",
        "--qat-steps", "100",
        "--output-dir", $OutputDir,
        "--work-dir", $WorkDir
    )

    if ($ExtraArgs) {
        $PyArgs += $ExtraArgs
    }

    Write-Log "Команда: python $([string]::Join(' ', $PyArgs))"

    # Фикс 7: Правильная обработка вывода и ошибок
    $ExitCode = 0
    $Output = @()
    
    try {
        $Output = & python @PyArgs 2>&1
        $ExitCode = $LASTEXITCODE
    }
    catch {
        $ExitCode = 1
        $Output = @("Ошибка запуска: $_")
    }

    # Записываем вывод в лог
    $Output | ForEach-Object { 
        $_ | Out-String | ForEach-Object { Add-Content -Path $Log -Value $_ -Encoding UTF8 }
    }
    # Показываем вывод на экран
    $Output | ForEach-Object { Write-Host $_ }

    $Elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds)

    if ($ExitCode -eq 0) {
        Write-Log "  OK (${Elapsed} сек)"
        $script:Succeeded++
        $script:Times += "$Pruning+$Method+$Order': ${Elapsed}s"
    }
    else {
        Write-Log "  FAIL (код: $ExitCode)"
        $script:Failed++
    }
}

# Запуск всех комбинаций
Run-One "magnitude_20" "ptq" "prune_first"
Run-One "magnitude_30" "ptq" "prune_first"
Run-One "structured"   "ptq" "prune_first"

Run-One "magnitude_20" "qat" "prune_first"
Run-One "magnitude_30" "qat" "prune_first"
Run-One "structured"   "qat" "prune_first"

Run-One "magnitude_20" "qat" "quant_first"
Run-One "magnitude_30" "qat" "quant_first"
Run-One "structured"   "qat" "quant_first"

Write-Log ""
Write-Log "Итого: $Succeeded/$Total успешно, $Failed ошибок"
Write-Log ""

foreach ($t in $Times) {
    Write-Log "  $t"
}

Write-Log ""
Write-Log "Готовые модели:"

# Фикс 8: Проверка существования файлов перед обработкой
if (Test-Path $OutputDir) {
    $GgufFiles = Get-ChildItem -Path $OutputDir -Filter "*.gguf" -ErrorAction SilentlyContinue
    if ($GgufFiles) {
        foreach ($f in $GgufFiles) {
            $Size = [math]::Round($f.Length / 1MB, 2)
            Write-Log "  ${Size}MB  $($f.Name)"
        }
    }
    else {
        Write-Log "  .gguf файлы не найдены"
    }
}
else {
    Write-Log "  Директория output не существует"
}

Write-Log ""
Write-Log "Отправьте на одноплатники: .\scripts\pc\deploy.ps1"
Write-Log "Лог: $Log"

# Фикс 9: Пауза в конце если запущено не из терминала
if ($Host.Name -eq 'ConsoleHost' -and -not $env:TERM) {
    Write-Host "`nНажмите Enter для закрытия..." -ForegroundColor Cyan
    Read-Host
}