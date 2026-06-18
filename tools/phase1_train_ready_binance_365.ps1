param(
    [int]$Days = 365,
    [string]$Step = "1h",
    [string]$Groups = "primary,trend_radar,daily",
    [string]$Phase1OutputDir = ".data\research\phase1_binance_365_dense",
    [string]$ModelOutputDir = ".data\research\phase3_entry_quality_binance_365_dense",
    [string]$SLTPOutputDir = ".data\research\sl_tp_analysis_binance_365_dense",
    [int]$MaxHoldBars = 96,
    [double]$EntryFillTolerancePct = 0.25,
    [int]$Workers = 0,
    [int]$ProgressEvery = 25,
    [int]$TestDays = 45,
    [int]$MinTrainDays = 120,
    [switch]$RefreshCache,
    [switch]$StrictCoverage
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:MARKET_DATA_PROVIDER = "binance"
$env:BINANCE_DEFAULT_QUOTE = "USDT"

$logDir = Join-Path $root ".data\research\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "phase1_train_ready_binance_365_$timestamp.log"
$phase1OutputAbs = if ([System.IO.Path]::IsPathRooted($Phase1OutputDir)) { $Phase1OutputDir } else { Join-Path $root $Phase1OutputDir }
$modelOutputAbs = if ([System.IO.Path]::IsPathRooted($ModelOutputDir)) { $ModelOutputDir } else { Join-Path $root $ModelOutputDir }
$sltpOutputAbs = if ([System.IO.Path]::IsPathRooted($SLTPOutputDir)) { $SLTPOutputDir } else { Join-Path $root $SLTPOutputDir }
$phase1DatasetPath = Join-Path $phase1OutputAbs "phase1_candidates.csv"

Start-Transcript -Path $logPath -Force | Out-Null

try {
    Write-Host "== Binance 365d Train-Ready =="
    Write-Host "MARKET_DATA_PROVIDER: $env:MARKET_DATA_PROVIDER"
    Write-Host "BINANCE_DEFAULT_QUOTE: $env:BINANCE_DEFAULT_QUOTE"
    Write-Host "Days: $Days"
    Write-Host "Step: $Step"
    Write-Host "Groups: $Groups"
    Write-Host "Max hold bars: $MaxHoldBars"
    Write-Host "Entry fill tolerance pct: $EntryFillTolerancePct"
    Write-Host "Workers: $Workers"
    Write-Host "Progress every: $ProgressEvery"
    Write-Host "Refresh market cache: $($RefreshCache.IsPresent)"
    Write-Host "Strict coverage: $($StrictCoverage.IsPresent)"
    Write-Host "Phase1 output: $phase1OutputAbs"
    Write-Host "Model output: $modelOutputAbs"
    Write-Host "SL/TP analysis output: $sltpOutputAbs"
    Write-Host "Log: $logPath"
    Write-Host ""

    $phase1Args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "daily_phase1_refresh_binance.ps1"),
        "-Days", $Days,
        "-Step", $Step,
        "-Groups", $Groups,
        "-OutputDir", $Phase1OutputDir,
        "-MaxHoldBars", $MaxHoldBars,
        "-EntryFillTolerancePct", $EntryFillTolerancePct,
        "-Workers", $Workers,
        "-ProgressEvery", $ProgressEvery
    )

    if (-not $RefreshCache.IsPresent) {
        $phase1Args += "-NoRefreshCache"
    }
    if ($StrictCoverage.IsPresent) {
        $phase1Args += "-NoAllowPartialCoverage"
    }

    Write-Host "== Step 1/3: Build Phase 1 Dataset =="
    & powershell @phase1Args
    $phase1ExitCode = $LASTEXITCODE
    if ($phase1ExitCode -ne 0) {
        throw "Phase 1 build failed with exit code $phase1ExitCode"
    }

    if (-not (Test-Path $phase1DatasetPath)) {
        throw "Phase 1 dataset not found after build: $phase1DatasetPath"
    }

    New-Item -ItemType Directory -Force -Path $sltpOutputAbs | Out-Null
    New-Item -ItemType Directory -Force -Path $modelOutputAbs | Out-Null

    Write-Host ""
    Write-Host "== Step 2/3: Analyze SL/TP Buy vs Sell =="
    & python ".\tools\analyze_phase1_sl_tp.py" `
        --input-path $phase1DatasetPath `
        --output-dir $sltpOutputAbs `
        --groups $Groups
    $analysisExitCode = $LASTEXITCODE
    if ($analysisExitCode -ne 0) {
        throw "SL/TP analysis failed with exit code $analysisExitCode"
    }

    Write-Host ""
    Write-Host "== Step 3/3: Train Model C =="
    & python ".\tools\train_phase3_entry_quality_model.py" `
        --input-path $phase1DatasetPath `
        --output-dir $modelOutputAbs `
        --groups $Groups `
        --test-days $TestDays `
        --min-train-days $MinTrainDays
    $trainExitCode = $LASTEXITCODE
    if ($trainExitCode -ne 0) {
        throw "Model C training failed with exit code $trainExitCode"
    }

    Write-Host ""
    Write-Host "Train-ready pipeline completed successfully." -ForegroundColor Green
    Write-Host "Phase1 dataset: $phase1DatasetPath"
    Write-Host "SL/TP analysis dir: $sltpOutputAbs"
    Write-Host "Model artifact dir: $modelOutputAbs"
    Write-Host "Run log: $logPath"
}
finally {
    Stop-Transcript | Out-Null
}
