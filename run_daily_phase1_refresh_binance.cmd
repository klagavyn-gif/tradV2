@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "MARKET_DATA_PROVIDER=binance"
set "BINANCE_DEFAULT_QUOTE=USDT"

echo ============================================
echo Daily Phase1 Refresh - Binance Local
echo ============================================
echo.
echo Provider: %MARKET_DATA_PROVIDER%
echo Quote: %BINANCE_DEFAULT_QUOTE%
echo.
echo This will:
echo 1. Refresh local market cache
echo 2. Use Binance as the local market data provider
echo 3. Rebuild Phase 1 dataset
echo 4. Save a log under .data\research\logs
echo.

powershell -ExecutionPolicy Bypass -File "%ROOT%tools\daily_phase1_refresh_binance.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Refresh failed with exit code %EXIT_CODE%.
) else (
    echo Refresh completed successfully.
)
echo.
pause
exit /b %EXIT_CODE%
