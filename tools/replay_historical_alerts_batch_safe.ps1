param(
    [int]$Days = 7,
    [string]$Step = "6h",
    [string]$Output = ".data/telegram_alerts/historical_replay_summary_7d_batch_safe.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$watchlist = @(
    "BTC-USD",
    "DOGE-USD",
    "ETH-USD",
    "ADA-USD",
    "XRP-USD",
    "BNB-USD",
    "SOL-USD",
    "TRX-USD",
    "NEAR-USD",
    "LINK-USD",
    "PAXG-USD"
)

function Write-Utf8NoBom {
    param(
        [string]$Path,
        [string]$Text
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Add-Count {
    param(
        [hashtable]$Map,
        [string]$Key,
        [int]$Delta = 1
    )

    if ([string]::IsNullOrWhiteSpace($Key)) {
        $Key = "UNKNOWN"
    }
    if ($Map.ContainsKey($Key)) {
        $Map[$Key] = [int]$Map[$Key] + $Delta
    } else {
        $Map[$Key] = $Delta
    }
}

function Add-MatrixCount {
    param(
        [hashtable]$Matrix,
        [string]$Strategy,
        [string]$Symbol,
        [int]$Delta = 1
    )

    if ([string]::IsNullOrWhiteSpace($Strategy)) {
        $Strategy = "UNKNOWN"
    }
    if ([string]::IsNullOrWhiteSpace($Symbol)) {
        $Symbol = "UNKNOWN"
    }
    if (-not $Matrix.ContainsKey($Strategy)) {
        $Matrix[$Strategy] = @{}
    }
    $row = [hashtable]$Matrix[$Strategy]
    if ($row.ContainsKey($Symbol)) {
        $row[$Symbol] = [int]$row[$Symbol] + $Delta
    } else {
        $row[$Symbol] = $Delta
    }
}

if ($Step -notmatch '^(\d+)([mhd])$') {
    throw "Unsupported step format: $Step"
}

$stepCount = [int]$Matches[1]
$stepUnit = $Matches[2]
switch ($stepUnit) {
    "m" { $interval = [TimeSpan]::FromMinutes($stepCount) }
    "h" { $interval = [TimeSpan]::FromHours($stepCount) }
    "d" { $interval = [TimeSpan]::FromDays($stepCount) }
    default { throw "Unsupported step unit: $stepUnit" }
}

$latestIso = & python -c "import pandas as pd; from pathlib import Path; from tools.replay_historical_alerts import load_cache, WATCHLIST; root = Path('.').resolve(); cache = load_cache(root, WATCHLIST); latest = min(df.index.max() for df in cache.values()); print(pd.Timestamp(latest).isoformat())"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to discover latest checkpoint timestamp"
}

$latest = [datetime]::Parse($latestIso.Trim())
$start = $latest.AddDays(-$Days)
$points = @()
for ($point = $start; $point -le $latest; $point = $point.Add($interval)) {
    $points += $point
}

$tempDir = Join-Path $root ".data/telegram_alerts/replay_batch_safe_tmp"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$byStrategy = @{}
$bySymbol = @{}
$bySignal = @{}
$byDay = @{}
$strategySymbolMatrix = @{}
$sampleAlerts = @()
$failedCheckpoints = @()
$dispatchDrops = @{ cache = 0; symbol_cap = 0; run_cap = 0 }
$totalAlerts = 0
$totalCandidateCount = 0
$runReportCount = 0

foreach ($symbol in $watchlist) {
    $safeSymbol = $symbol -replace '[^A-Za-z0-9._-]', '_'
    $statePath = Join-Path $tempDir "${safeSymbol}_state.json"
    $resultPath = Join-Path $tempDir "${safeSymbol}_result.json"
    $statePathAbs = [System.IO.Path]::GetFullPath($statePath)
    $resultPathAbs = [System.IO.Path]::GetFullPath($resultPath)
    Write-Utf8NoBom -Path $statePathAbs -Text (@{ alert_cache = @() } | ConvertTo-Json -Depth 10)

    foreach ($point in $points) {
        if (Test-Path $resultPathAbs) {
            Remove-Item -Path $resultPathAbs -Force
        }

        $workerOutput = & python tools/replay_historical_alerts.py `
            --mode checkpoint `
            --checkpoint-at $point.ToString("s") `
            --watchlist $symbol `
            --state-input $statePathAbs `
            --result-output $resultPathAbs 2>&1

        $exitCode = $LASTEXITCODE
        if (-not (Test-Path $resultPathAbs)) {
            $failedCheckpoints += [pscustomobject]@{
                symbol = $symbol
                timestamp = $point.ToString("yyyy-MM-dd HH:mm:ss")
                returncode = $exitCode
                worker_output = ($workerOutput | Out-String).Trim()
            }
            continue
        }

        $result = Get-Content -Path $resultPathAbs -Raw | ConvertFrom-Json
        Write-Utf8NoBom -Path $statePathAbs -Text (@{ alert_cache = @($result.alert_cache) } | ConvertTo-Json -Depth 20)

        foreach ($alert in @($result.alerts)) {
            $totalAlerts += 1
            Add-Count -Map $byStrategy -Key ([string]$alert.strategy)
            Add-Count -Map $bySymbol -Key ([string]$alert.symbol)
            Add-Count -Map $bySignal -Key ([string]$alert.signal)
            $dayText = ([string]$alert.timestamp).Substring(0, 10)
            Add-Count -Map $byDay -Key $dayText
            Add-MatrixCount -Matrix $strategySymbolMatrix -Strategy ([string]$alert.strategy) -Symbol ([string]$alert.symbol)
            if ($sampleAlerts.Count -lt 10) {
                $sampleAlerts += $alert
            }
        }

        foreach ($report in @($result.run_reports)) {
            $runReportCount += 1
            $totalCandidateCount += [int]$report.candidate_count
            $dispatchDrops["cache"] += [int]$report.dropped_by_cache
            $dispatchDrops["symbol_cap"] += [int]$report.dropped_by_symbol_cap
            $dispatchDrops["run_cap"] += [int]$report.dropped_by_run_cap
        }
    }
}

$outputPath = $Output
if (-not [System.IO.Path]::IsPathRooted($outputPath)) {
    $outputPath = Join-Path $root $Output
}

$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$summary = [ordered]@{
    mode = "batch-safe-per-symbol"
    window_days = $Days
    step = $Step
    checkpoints = $points.Count
    watchlist = $watchlist
    symbols_count = $watchlist.Count
    total_symbol_checkpoints = $points.Count * $watchlist.Count
    total_alerts = $totalAlerts
    alerts_per_day_avg = if ($Days) { [math]::Round($totalAlerts / [double]$Days, 4) } else { 0.0 }
    by_strategy = $byStrategy
    by_symbol = $bySymbol
    by_signal = $bySignal
    by_day = $byDay
    strategy_symbol_matrix = $strategySymbolMatrix
    avg_candidates_per_run = if ($runReportCount) { [math]::Round($totalCandidateCount / [double]$runReportCount, 4) } else { 0.0 }
    dispatch_drops = $dispatchDrops
    failed_checkpoints = $failedCheckpoints
    failed_checkpoints_count = @($failedCheckpoints).Count
    sample_alerts = $sampleAlerts
}

$summaryJson = $summary | ConvertTo-Json -Depth 20
Write-Utf8NoBom -Path $outputPath -Text $summaryJson
$summaryJson
