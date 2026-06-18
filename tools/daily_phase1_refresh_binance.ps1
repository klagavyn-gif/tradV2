param(
    [int]$Days = 90,
    [string]$Step = "4h",
    [string]$Groups = "primary,daily",
    [string]$OutputDir = ".data\research\phase1",
    [int]$MaxHoldBars = 64,
    [double]$EntryFillTolerancePct = 0.15,
    [int]$Workers = 0,
    [int]$ProgressEvery = 25,
    [switch]$NoRefreshCache,
    [switch]$NoAllowPartialCoverage
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:MARKET_DATA_PROVIDER = "binance"
$env:BINANCE_DEFAULT_QUOTE = "USDT"

Write-Host "== Daily Phase1 Refresh (Binance Local) =="
Write-Host "MARKET_DATA_PROVIDER: $env:MARKET_DATA_PROVIDER"
Write-Host "BINANCE_DEFAULT_QUOTE: $env:BINANCE_DEFAULT_QUOTE"
Write-Host ""

$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "daily_phase1_refresh.ps1"),
    "-Days", $Days,
    "-Step", $Step,
    "-Groups", $Groups,
    "-OutputDir", $OutputDir,
    "-MaxHoldBars", $MaxHoldBars,
    "-EntryFillTolerancePct", $EntryFillTolerancePct,
    "-Workers", $Workers,
    "-ProgressEvery", $ProgressEvery
)

if ($NoRefreshCache.IsPresent) {
    $arguments += "-NoRefreshCache"
}
if ($NoAllowPartialCoverage.IsPresent) {
    $arguments += "-NoAllowPartialCoverage"
}

& powershell @arguments
exit $LASTEXITCODE
