param(
    [int]$Days = 90,
    [string]$Step = "4h",
    [string]$Groups = "primary,daily",
    [string]$OutputDir = ".data\research\phase1",
    [switch]$NoRefreshCache,
    [switch]$NoAllowPartialCoverage
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$refreshCache = -not $NoRefreshCache.IsPresent
$allowPartialCoverage = -not $NoAllowPartialCoverage.IsPresent

$outputDirAbs = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $root $OutputDir
}

New-Item -ItemType Directory -Force -Path $outputDirAbs | Out-Null

$logDir = Join-Path $root ".data\research\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "phase1_refresh_$timestamp.log"

$arguments = @(
    ".\tools\build_phase1_research_dataset.py"
    "--days", $Days
    "--step", $Step
    "--groups", $Groups
    "--output-dir", $OutputDir
)

if ($refreshCache) {
    $arguments += "--refresh-cache"
}
if ($allowPartialCoverage) {
    $arguments += "--allow-partial-coverage"
}

Write-Host "== Daily Phase1 Refresh =="
Write-Host "Root: $root"
Write-Host "Days: $Days"
Write-Host "Step: $Step"
Write-Host "Groups: $Groups"
Write-Host "Output: $OutputDirAbs"
Write-Host "Refresh cache: $refreshCache"
Write-Host "Allow partial coverage: $allowPartialCoverage"
Write-Host "Log: $logPath"
Write-Host ""

$output = & python @arguments 2>&1
$exitCode = $LASTEXITCODE

$outputText = ($output | Out-String)
$outputText | Tee-Object -FilePath $logPath

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "Phase 1 refresh failed. See log: $logPath" -ForegroundColor Red
    exit $exitCode
}

Write-Host ""
Write-Host "Phase 1 refresh completed successfully." -ForegroundColor Green
Write-Host "Files should be available under: $outputDirAbs"
Write-Host "Recommended next check:"
Write-Host "  Get-ChildItem `"$OutputDir`""
