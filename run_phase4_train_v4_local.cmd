@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ============================================
echo Local Train V4 UTF Prototype
echo ============================================
echo.
echo This will:
echo 1. Load a Phase 1 dataset from local research output
echo 2. Build V4 composite features and schema fields
echo 3. Train the V4 UTF prototype with auto backend
echo 4. Scan Premium / Standard / Watch policies
echo 5. Show Step 1/5 ... Step 5/5 progress
echo 6. Show %% and ETA during policy scan
echo 7. Save artifacts under .data\research\phase4_entry_quality_v4_local
echo.
echo Notes:
echo - Default backend is auto
echo - If xgboost is not installed, it falls back to CPU
echo - Default input is .data\research\phase1_binance_365_dense\phase1_candidates.csv
echo.

powershell -ExecutionPolicy Bypass -File "%ROOT%tools\phase4_train_v4_local.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Train V4 failed with exit code %EXIT_CODE%.
) else (
    echo Train V4 completed successfully.
)
echo.
pause
exit /b %EXIT_CODE%
