@echo off
setlocal
cd /d "%~dp0"
python tools\report_realized_alerts.py %*
echo.
pause
