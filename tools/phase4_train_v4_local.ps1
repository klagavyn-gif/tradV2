param(
    [string]$InputPath = ".data\research\phase1_binance_365_dense\phase1_candidates.csv",
    [string]$OutputDir = ".data\research\phase4_entry_quality_v4_local",
    [string]$Groups = "primary",
    [string]$Strategies = "",
    [int]$TestDays = 45,
    [int]$MinTrainDays = 120,
    [int]$MinClassRows = 20,
    [string]$Backend = "auto",
    [string]$Device = "auto",
    [string]$CalibrationMethod = "platt",
    [int]$CalibrationDays = 21,
    [int]$CalibrationMinRows = 60,
    [int]$CalibrationMinTrainDays = 60,
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
    [double]$MinAvgReturnPct = 2.0,
    [int]$PremiumMinSelectedRows = 40,
    [double]$PremiumTargetAlertsPerDay = 1.0,
    [double]$PremiumMaxAlertsPerDay = 1.5,
    [double]$PremiumMinWinRatePct = 60.0,
    [double]$PremiumMaxWinRatePct = 65.0,
    [double]$PremiumMinAvgReturnPct = 2.0,
    [int]$StandardMinSelectedRows = 80,
    [double]$StandardTargetAlertsPerDay = 2.5,
    [double]$StandardMaxAlertsPerDay = 4.0,
    [double]$StandardMinWinRatePct = 55.0,
    [double]$StandardMaxWinRatePct = 60.0,
    [double]$StandardMinAvgReturnPct = 1.5,
    [int]$WatchMinSelectedRows = 0,
    [double]$WatchTargetAlertsPerDay = 3.0,
    [double]$WatchMaxAlertsPerDay = 6.0,
    [double]$WatchMinWinRatePct = 48.0,
    [double]$WatchMaxWinRatePct = 57.5,
    [double]$WatchMinAvgReturnPct = 1.0,
    [double]$WatchEntryThresholdMin = 0.50,
    [double]$WatchEntryThresholdMax = 0.80,
    [double]$WatchEntryThresholdStep = 0.05,
    [double]$WatchAvoidThresholdMin = 0.70,
    [double]$WatchAvoidThresholdMax = 0.95,
    [double]$WatchAvoidThresholdStep = 0.05,
    [double]$WatchMinMasterScore = 0.19,
    [double]$WatchMinRegimeScore = 0.58,
    [double]$WatchMinDirectionScore = 0.46,
    [double]$WatchMinEntryPrecisionScore = 0.18,
    [double]$WatchMinExitQualityScore = 0.34,
    [double]$WatchMinExecutionUtilityScore = 0.40,
    [double]$PolicyCalibrationTargetWeight = 0.10,
    [double]$PolicyCalibrationAvoidWeight = 0.06,
    [double]$PolicyCalibrationOverconfidencePenaltyWeight = 0.08,
    [int]$StrategyPolicyMinHoldoutRows = 90
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root ".data\research\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "phase4_train_v4_local_$timestamp.log"
$outputAbs = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $root $OutputDir }
$inputAbs = if ([System.IO.Path]::IsPathRooted($InputPath)) { $InputPath } else { Join-Path $root $InputPath }

Start-Transcript -Path $logPath -Force | Out-Null

try {
    Write-Host "== Local Train V4 UTF Prototype =="
    Write-Host "Input: $inputAbs"
    Write-Host "Output: $outputAbs"
    Write-Host "Groups: $Groups"
    Write-Host "Strategies: $Strategies"
    Write-Host "Backend: $Backend"
    Write-Host "Device: $Device"
    Write-Host "Calibration: $CalibrationMethod"
    Write-Host "Holdout days: $TestDays"
    Write-Host "Min train days: $MinTrainDays"
    Write-Host "Strategy policy min holdout rows: $StrategyPolicyMinHoldoutRows"
    Write-Host "Premium target alerts/day: $PremiumTargetAlertsPerDay"
    Write-Host "Standard target alerts/day: $StandardTargetAlertsPerDay"
    Write-Host "Watch target alerts/day: $WatchTargetAlertsPerDay"
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
        "--calibration-method", $CalibrationMethod,
        "--calibration-days", $CalibrationDays,
        "--calibration-min-rows", $CalibrationMinRows,
        "--calibration-min-train-days", $CalibrationMinTrainDays,
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
        "--min-avg-return-pct", $MinAvgReturnPct,
        "--premium-min-selected-rows", $PremiumMinSelectedRows,
        "--premium-target-alerts-per-day", $PremiumTargetAlertsPerDay,
        "--premium-max-alerts-per-day", $PremiumMaxAlertsPerDay,
        "--premium-min-win-rate-pct", $PremiumMinWinRatePct,
        "--premium-max-win-rate-pct", $PremiumMaxWinRatePct,
        "--premium-min-avg-return-pct", $PremiumMinAvgReturnPct,
        "--standard-min-selected-rows", $StandardMinSelectedRows,
        "--standard-target-alerts-per-day", $StandardTargetAlertsPerDay,
        "--standard-max-alerts-per-day", $StandardMaxAlertsPerDay,
        "--standard-min-win-rate-pct", $StandardMinWinRatePct,
        "--standard-max-win-rate-pct", $StandardMaxWinRatePct,
        "--standard-min-avg-return-pct", $StandardMinAvgReturnPct,
        "--watch-min-selected-rows", $WatchMinSelectedRows,
        "--watch-target-alerts-per-day", $WatchTargetAlertsPerDay,
        "--watch-max-alerts-per-day", $WatchMaxAlertsPerDay,
        "--watch-min-win-rate-pct", $WatchMinWinRatePct,
        "--watch-max-win-rate-pct", $WatchMaxWinRatePct,
        "--watch-min-avg-return-pct", $WatchMinAvgReturnPct,
        "--watch-entry-threshold-min", $WatchEntryThresholdMin,
        "--watch-entry-threshold-max", $WatchEntryThresholdMax,
        "--watch-entry-threshold-step", $WatchEntryThresholdStep,
        "--watch-avoid-threshold-min", $WatchAvoidThresholdMin,
        "--watch-avoid-threshold-max", $WatchAvoidThresholdMax,
        "--watch-avoid-threshold-step", $WatchAvoidThresholdStep,
        "--watch-min-master-score", $WatchMinMasterScore,
        "--watch-min-regime-score", $WatchMinRegimeScore,
        "--watch-min-direction-score", $WatchMinDirectionScore,
        "--watch-min-entry-precision-score", $WatchMinEntryPrecisionScore,
        "--watch-min-exit-quality-score", $WatchMinExitQualityScore,
        "--watch-min-execution-utility-score", $WatchMinExecutionUtilityScore,
        "--policy-calibration-target-weight", $PolicyCalibrationTargetWeight,
        "--policy-calibration-avoid-weight", $PolicyCalibrationAvoidWeight,
        "--policy-calibration-overconfidence-penalty-weight", $PolicyCalibrationOverconfidencePenaltyWeight,
        "--strategy-policy-enable",
        "--strategy-policy-min-holdout-rows", $StrategyPolicyMinHoldoutRows
    )
    if (-not [string]::IsNullOrWhiteSpace($Strategies)) {
        $pythonArgs += @("--strategies", $Strategies)
    }

    & python @pythonArgs

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Train V4 failed with exit code $exitCode"
    }

    Write-Host ""
    Write-Host "Train V4 completed successfully." -ForegroundColor Green
    Write-Host "Output dir: $outputAbs"
    Write-Host "Run log: $logPath"
}
finally {
    Stop-Transcript | Out-Null
}
