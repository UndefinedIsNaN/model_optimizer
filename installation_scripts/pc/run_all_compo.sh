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

$ScriptDir = $PSScriptRoot
$ProjectDir = (Resolve-Path "$ScriptDir\..\..").Path

# Активация виртуального окружения
& "$ProjectDir\venv\Scripts\Activate.ps1"
Set-Location $ProjectDir

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = "$ProjectDir\logs\all_combos_$Timestamp.log"
New-Item -ItemType Directory -Force -Path "$ProjectDir\logs" | Out-Null

function Write-Log {
    param([string]$Message)
    $Message | Tee-Object -FilePath $Log -Append
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

    $ArgsList = @(
        "main.py",
        "--repo", $Repo,
        "--pruning", $Pruning,
        "--quant-method", $Method,
        "--quant-type", $QuantType,
        "--order", $Order,
        "--recovery-steps", "50",
        "--qat-steps", "100",
        "--output-dir", "$ProjectDir\output",
        "--work-dir", "$ProjectDir\work"
    ) + $ExtraArgs

    try {
        & python @ArgsList 2>&1 | Tee-Object -FilePath $Log -Append
        $ExitCode = $LASTEXITCODE
    }
    catch {
        $ExitCode = 1
    }

    $Elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds)

    if ($ExitCode -eq 0) {
        Write-Log "  OK (${Elapsed} сек)"
        $script:Succeeded++
        $script:Times += "$Pruning+$Method+$Order: ${Elapsed}s"
    }
    else {
        Write-Log "  FAIL"
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

$GgufFiles = Get-ChildItem -Path "$ProjectDir\output" -Filter "*.gguf" -ErrorAction SilentlyContinue
foreach ($f in $GgufFiles) {
    $Size = [math]::Round($f.Length / 1MB, 2)
    Write-Log "  ${Size}MB  $($f.Name)"
}

Write-Log ""
Write-Log "Отправьте на одноплатники: .\scripts\pc\deploy.ps1"
Write-Log "Лог: $Log"