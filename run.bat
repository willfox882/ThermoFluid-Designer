@echo off
REM ThermoFluid Designer - Windows launcher
REM Run this from the thermofluid_designer\ directory

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Check dependencies
python -c "import PyQt6, numpy, scipy, matplotlib" >nul 2>&1
IF ERRORLEVEL 1 (
    echo Installing required packages...
    pip install PyQt6 numpy scipy matplotlib
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to install packages.
        pause
        exit /b 1
    )
)

echo Starting ThermoFluid Designer...
python main.py
IF ERRORLEVEL 1 (
    echo.
    echo Application exited with an error.
    pause
)
