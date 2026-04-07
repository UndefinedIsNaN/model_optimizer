<#
Запуск всех комбинаций квантизации
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Repo,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Continue"

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$ProjectDir = Resolve-Path "$ScriptDir\..\.." | Select-Object -ExpandProperty Path

# Активация venv
$VenvPath = "$ProjectDir\venv\Scripts\Activate.ps1"
& $VenvPath

Set-Location $ProjectDir

# Создание директорий
$LogDir = "$ProjectDir\logs"
$OutputDir = "$ProjectDir\output_quant"
@($LogDir, $OutputDir) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ -ErrorAction SilentlyContinue | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = "$LogDir\all_quantizations_$Timestamp.log"

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

Write-Log "Запуск всех комбинаций квантизации" "Green"
Write-Log "  Repo: $Repo"
Write-Log "  Python: $(python --version 2>&1)"
Write-Log ""

$PTQ_TYPES = @("Q2_K", "Q3_K_S", "Q4_K_S", "Q4_K_M", "Q6_K", "Q8_0")
$QAT_TYPES = @("Q2_K", "Q4_K_S", "Q8_0")

$Total = $PTQ_TYPES.Count + $QAT_TYPES.Count
$Current = 0
$Failed = 0
$Succeeded = 0
$Times = @()

function Run-Quantization {
    param([string]$Method, [string]$QuantType)
    
    $script:Current++
    Write-Log ""
    Write-Log "[$Current/$Total] $Method + $QuantType" "Cyan"

    $StartTime = Get-Date
    $TempLog = "$LogDir\temp_$Method`_$QuantType`_$Timestamp.log"

    $PyArgs = @(
        "$ProjectDir\Quant.py",
        "--repo", $Repo,
        "--quant-method", $Method,
        "--quant-type", $QuantType,
        "--output-dir", $OutputDir
    )
    if ($ExtraArgs) { $PyArgs += $ExtraArgs }

    Write-Log "Запуск: python Quant.py --repo $Repo --quant-method $Method --quant-type $QuantType"

    # Запуск через Start-Process с логированием в файл (не блокирует)
    $process = $null
    
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "python"
        $psi.Arguments = [string]::Join(" ", $PyArgs)
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.WorkingDirectory = $ProjectDir
        # Не перенаправляем потоки - пусть пишет в консоль и файл через tee
        $psi.RedirectStandardOutput = $false
        $psi.RedirectStandardError = $false
        
        $process = [System.Diagnostics.Process]::Start($psi)
        
        Write-Log "PID: $($process.Id), ожидание завершения..."
        
        # Ожидание с таймаутом и прогрессом
        $timeout = 3600  # 1 час максимум
        $elapsed = 0
        
        while (-not $process.HasExited -and $elapsed -lt $timeout) {
            Start-Sleep -Seconds 5
            $elapsed += 5
            $mins = [math]::Floor($elapsed / 60)
            $secs = $elapsed % 60
            Write-Host "`r  Работает... ${mins}m ${secs}s" -NoNewline -ForegroundColor DarkGray
        }
        Write-Host ""  # Новая строка
        
        if (-not $process.HasExited) {
            $process.Kill()
            throw "Таймаут выполнения (> 1 часа)"
        }
        
        $ExitCode = $process.ExitCode
    }
    catch {
        $ExitCode = 999
        Write-Log "Исключение: $_" "Red"
    }
    finally {
        if ($process) { $process.Dispose() }
    }

    $Elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds)

    if ($ExitCode -eq 0) {
        Write-Log "  OK (${Elapsed} сек)" "Green"
        $script:Succeeded++
        $script:Times += "$Method+$QuantType`: ${Elapsed}s"
    }
    else {
        Write-Log "  FAIL (код: $ExitCode, ${Elapsed} сек)" "Red"
        $script:Failed++
    }
}

# PTQ
Write-Log "━" * 60 "Gray"
Write-Log "PTQ (Post-Training Quantization)" "Yellow"
Write-Log "━" * 60 "Gray"

foreach ($qt in $PTQ_TYPES) {
    Run-Quantization "ptq" $qt
}

# QAT
Write-Log ""
Write-Log "━" * 60 "Gray"
Write-Log "QAT (Quantization-Aware Training)" "Yellow"
Write-Log "━" * 60 "Gray"

foreach ($qt in $QAT_TYPES) {
    Run-Quantization "qat" $qt
}

# Итоги
Write-Log ""
Write-Log "━" * 60 "Gray"
Write-Log "ИТОГИ: $Succeeded/$Total успешно, $Failed ошибок" $(if($Failed -gt 0){"Red"}else{"Green"})
Write-Log "━" * 60 "Gray"

foreach ($t in $Times) {
    Write-Log "  $t"
}

Write-Log ""
Write-Log "Готовые модели:"
$GgufFiles = Get-ChildItem -Path $OutputDir -Filter "*.gguf" -ErrorAction SilentlyContinue | Sort-Object Length
if ($GgufFiles) {
    foreach ($f in $GgufFiles) {
        $SizeMB = [math]::Round($f.Length / 1MB, 2)
        Write-Log ("  {0,6:F1} MB  {1}" -f $SizeMB, $f.Name)
    }
}
else {
    Write-Log "  Нет .gguf файлов" "Yellow"
}

Write-Log ""
Write-Log "Лог сохранён: $Log" "Gray"

# Пауза
if ($Host.Name -eq 'ConsoleHost') {
    Write-Host "`nНажмите Enter для закрытия..." -ForegroundColor Cyan
    Read-Host
}