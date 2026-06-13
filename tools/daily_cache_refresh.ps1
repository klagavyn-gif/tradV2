param(
    [int]$Days = 90,
    [string]$Watchlist = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root ".data\research\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "cache_refresh_$timestamp.log"
$tempPy = Join-Path ([System.IO.Path]::GetTempPath()) "tradv2_cache_refresh_$timestamp.py"

$pythonScript = @'
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    days = int(sys.argv[2])
    watchlist_arg = sys.argv[3] if len(sys.argv) > 3 else ""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from tools.build_phase1_research_dataset import (
        load_cache,
        parse_watchlist,
        refresh_cache,
        summarize_cache_coverage,
    )
    import trad

    watchlist = parse_watchlist(watchlist_arg)
    refreshed = refresh_cache(root, watchlist, days)
    cache = load_cache(root, watchlist)
    coverage = summarize_cache_coverage(cache, days=days)

    payload = {
        "mode": "cache_only_refresh",
        "root": str(root),
        "provider": trad.get_market_data_provider(),
        "history_store_dir": trad.get_market_history_store_dir(),
        "days": days,
        "watchlist_count": len(watchlist),
        "watchlist": watchlist,
        "refreshed": refreshed,
        "coverage": {
            "requested_days": coverage.get("requested_days"),
            "available_days": coverage.get("available_days"),
            "requested_start": coverage.get("requested_start"),
            "effective_start": coverage.get("effective_start"),
            "latest_end": coverage.get("latest_end"),
            "has_full_coverage": coverage.get("has_full_coverage"),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
'@

Set-Content -Path $tempPy -Value $pythonScript -Encoding UTF8

$provider = & python -c 'import sys; from pathlib import Path; root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root)); import trad; print(trad.get_market_data_provider())' $root

Write-Host "== Daily Cache Refresh =="
Write-Host "Root: $root"
Write-Host "Days: $Days"
Write-Host "Watchlist: $(if ([string]::IsNullOrWhiteSpace($Watchlist)) { '<default>' } else { $Watchlist })"
Write-Host "Log: $logPath"
Write-Host "Provider: $provider"
Write-Host ""

try {
    $pythonArgs = @($tempPy, $root, $Days)
    if (-not [string]::IsNullOrWhiteSpace($Watchlist)) {
        $pythonArgs += $Watchlist
    }
    $output = & python @pythonArgs 2>&1
    $exitCode = $LASTEXITCODE
    $outputText = ($output | Out-String)
    $outputText | Tee-Object -FilePath $logPath

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "Cache refresh failed. See log: $logPath" -ForegroundColor Red
        exit $exitCode
    }

    Write-Host ""
    Write-Host "Cache refresh completed successfully." -ForegroundColor Green
    $storeDir = & python -c 'import sys; from pathlib import Path; root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root)); import trad; print(trad.get_market_history_store_dir())' $root
    Write-Host "Market cache should now be updated under: $storeDir"
    Write-Host "Recommended next check:"
    Write-Host "  Get-ChildItem `"$storeDir`" | Sort-Object LastWriteTime -Descending | Select-Object -First 10"
}
finally {
    Remove-Item -Path $tempPy -ErrorAction SilentlyContinue
}
