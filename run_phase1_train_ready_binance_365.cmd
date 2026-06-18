@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "MARKET_DATA_PROVIDER=binance"
set "BINANCE_DEFAULT_QUOTE=USDT"

echo ============================================
echo Binance 365d Train-Ready
echo ============================================
echo.
echo Provider: %MARKET_DATA_PROVIDER%
echo Quote: %BINANCE_DEFAULT_QUOTE%
echo.
echo This will:
echo 1. Build a dense Phase 1 dataset from local Binance cache
echo 2. Use 365 days as the replay window with denser defaults
echo 3. Analyze SL/TP quality for BUY vs SELL
echo 4. Train Model C from the same dataset
echo 5. Save a combined log under .data\research\logs
echo.
echo Note:
echo - Default mode uses existing local cache and skips cache refresh
echo - Default groups are primary,trend_radar,daily
echo - To refresh cache first, run the PowerShell script directly with -RefreshCache
echo.

powershell -ExecutionPolicy Bypass -File "%ROOT%tools\phase1_train_ready_binance_365.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Train-ready pipeline failed with exit code %EXIT_CODE%.
) else (
    echo Train-ready pipeline completed successfully.
)
echo.
pause
exit /b %EXIT_CODE%
