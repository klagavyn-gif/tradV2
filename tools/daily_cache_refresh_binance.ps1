param(
    [int]$Days = 90,
    [string]$Watchlist = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:MARKET_DATA_PROVIDER = "binance"
$env:BINANCE_DEFAULT_QUOTE = "USDT"

Write-Host "== Daily Cache Refresh (Binance Local) =="
Write-Host "MARKET_DATA_PROVIDER: $env:MARKET_DATA_PROVIDER"
Write-Host "BINANCE_DEFAULT_QUOTE: $env:BINANCE_DEFAULT_QUOTE"
Write-Host ""

$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "daily_cache_refresh.ps1"),
    "-Days", $Days
)

if (-not [string]::IsNullOrWhiteSpace($Watchlist)) {
    $arguments += @("-Watchlist", $Watchlist)
}

& powershell @arguments
exit $LASTEXITCODE
