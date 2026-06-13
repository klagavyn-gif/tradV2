@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ============================================
echo Daily Cache Refresh
echo ============================================
echo.
echo This will:
echo 1. Refresh local market cache only
echo 2. Skip rebuilding Phase 1 dataset
echo 3. Save a log under .data\research\logs
echo.

powershell -ExecutionPolicy Bypass -File "%ROOT%tools\daily_cache_refresh.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Cache refresh failed with exit code %EXIT_CODE%.
) else (
    echo Cache refresh completed successfully.
)
echo.
pause
exit /b %EXIT_CODE%
