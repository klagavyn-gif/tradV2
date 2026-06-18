param(
    [string]$InputPath = ".data\research\phase1_binance_365_dense\phase1_candidates.csv",
    [string]$OutputDir = ".data\research\phase3_entry_quality_v3_local",
    [string]$Groups = "primary",
    [string]$Strategies = "",
    [int]$TestDays = 45,
    [int]$MinTrainDays = 120,
    [int]$MinClassRows = 20,
    [string]$Backend = "auto",
    [string]$Device = "auto",
    [double]$EntryThresholdMin = 0.65,
    [double]$EntryThresholdMax = 0.95,
    [double]$EntryThresholdStep = 0.05,
    [double]$AvoidThresholdMin = 0.75,
    [double]$AvoidThresholdMax = 0.95,
    [double]$AvoidThresholdStep = 0.05,
    [int]$MinSelectedRows = 80,
    [double]$MinAlertsPerDay = 0.25,
    [double]$TargetAlertsPerDay = 1.0,
    [double]$MaxAlertsPerDay = 2.5,
    [double]$MinWinRatePct = 57.5,
    [double]$MinAvgReturnPct = 2.0
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root ".data\research\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "phase3_train_v3_local_$timestamp.log"
$outputAbs = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $root $OutputDir }
$inputAbs = if ([System.IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { Join-Path $root $InputPath }

Start-Transcript -Path $logPath -Force | Out-Null

try {
    Write-Host "== Local Train V3 =="
    Write-Host "Input: $inputAbs"
    Write-Host "Output: $outputAbs"
    Write-Host "Groups: $Groups"
    Write-Host "Strategies: $Strategies"
    Write-Host "Backend: $Backend"
    Write-Host "Device: $Device"
    Write-Host "Holdout days: $TestDays"
    Write-Host "Min train days: $MinTrainDays"
    Write-Host "Target alerts/day: $TargetAlertsPerDay"
    Write-Host "Max alerts/day: $MaxAlertsPerDay"
    Write-Host "Min win rate pct: $MinWinRatePct"
    Write-Host "Min avg return pct: $MinAvgReturnPct"
    Write-Host "Log: $logPath"
    Write-Host ""

    New-Item -ItemType Directory -Force -Path $outputAbs | Out-Null

    $pythonArgs = @(
        ".\tools\train_phase3_entry_quality_model_v3.py",
        "--input-path", $InputPath,
        "--output-dir", $OutputDir,
        "--groups", $Groups,
        "--test-days", $TestDays,
        "--min-train-days", $MinTrainDays,
        "--min-class-rows", $MinClassRows,
        "--backend", $Backend,
        "--device", $Device,
        "--entry-threshold-min", $EntryThresholdMin,
        "--entry-threshold-max", $EntryThresholdMax,
        "--entry-threshold-step", $EntryThresholdStep,
        "--avoid-threshold-min", $AvoidThresholdMin,
        "--avoid-threshold-max", $AvoidThresholdMax,
        "--avoid-threshold-step", $AvoidThresholdStep,
        "--min-selected-rows", $MinSelectedRows,
        "--min-alerts-per-day", $MinAlertsPerDay,
        "--target-alerts-per-day", $TargetAlertsPerDay,
        "--max-alerts-per-day", $MaxAlertsPerDay,
        "--min-win-rate-pct", $MinWinRatePct,
        "--min-avg-return-pct", $MinAvgReturnPct
    )
    if (-not [string]::IsNullOrWhiteSpace($Strategies)) {
        $pythonArgs += @("--strategies", $Strategies)
    }

    & python @pythonArgs

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Train V3 failed with exit code $exitCode"
    }

    Write-Host ""
    Write-Host "Train V3 completed successfully." -ForegroundColor Green
    Write-Host "Output dir: $outputAbs"
    Write-Host "Run log: $logPath"
}
finally {
    Stop-Transcript | Out-Null
}
