@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ============================================
echo Local Train V3
echo ============================================
echo.
echo This will:
echo 1. Load a Phase 1 dataset from local research output
echo 2. Build V3 composite features
echo 3. Train Model V3 with auto backend
echo 4. Show Step 1/5 ... Step 5/5 progress
echo 5. Show %% and ETA during threshold scan
echo 6. Save artifacts under .data\research\phase3_entry_quality_v3_local
echo.
echo Notes:
echo - Default backend is auto
echo - If xgboost is not installed, it falls back to CPU
echo - Default input is .data\research\phase1_binance_365_dense\phase1_candidates.csv
echo.

powershell -ExecutionPolicy Bypass -File "%ROOT%tools\phase3_train_v3_local.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Train V3 failed with exit code %EXIT_CODE%.
) else (
    echo Train V3 completed successfully.
)
echo.
pause
exit /b %EXIT_CODE%
