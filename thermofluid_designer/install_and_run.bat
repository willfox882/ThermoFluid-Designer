@echo off
title ThermoFluid Designer — Launcher
color 0B
echo.
echo  ============================================
echo   ThermoFluid Designer — Setup ^& Launch
echo  ============================================
echo.

REM ── Step 1: Check if Python is installed ────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Python is not installed or not on PATH.
    echo.
    echo  Please install Python 3.10 or later from:
    echo     https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During install, check the box:
    echo     [x] Add Python to PATH
    echo.
    echo  Then re-run this script.
    echo.
    pause
    exit /b 1
)

REM ── Step 2: Check Python version ────────────────────────────────────────
echo  [1/3] Checking Python version...
python -c "import sys; v=sys.version_info; exit(0 if v>=(3,10) else 1)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo  [WARNING] Python 3.10+ recommended. You may experience issues.
)
python --version
echo.

REM ── Step 3: Install dependencies ────────────────────────────────────────
echo  [2/3] Installing dependencies (PyQt6, NumPy, SciPy, Matplotlib)...
echo         This may take a minute on first run...
echo.
python -m pip install --upgrade pip >nul 2>&1
python -m pip install PyQt6>=6.4 numpy>=1.24 scipy>=1.10 matplotlib>=3.7 --quiet
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Failed to install dependencies.
    echo  Try running manually:
    echo     pip install PyQt6 numpy scipy matplotlib
    echo.
    pause
    exit /b 1
)
echo  Dependencies installed successfully.
echo.

REM ── Step 4: Launch the application ──────────────────────────────────────
echo  [3/3] Launching ThermoFluid Designer...
echo.
cd /d "%~dp0"
python main.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo  ============================================
    echo   Application exited with an error.
    echo   Error code: %ERRORLEVEL%
    echo  ============================================
    echo.
    echo  Common fixes:
    echo    1. Make sure all files are extracted (not run from inside .zip)
    echo    2. Try: pip install --force-reinstall PyQt6
    echo    3. Check that all .py files are in the same folder as main.py
    echo.
    pause
    exit /b 1
)
