@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ============================================
echo Daily Phase1 Refresh
echo ============================================
echo.
echo This will:
echo 1. Refresh local cache
echo 2. Rebuild Phase 1 dataset
echo 3. Save a log under .data\research\logs
echo.

powershell -ExecutionPolicy Bypass -File "%ROOT%tools\daily_phase1_refresh.ps1"
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
