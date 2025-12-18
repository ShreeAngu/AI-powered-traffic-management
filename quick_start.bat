@echo off
title AI Traffic Light System - Quick Start
echo ================================================
echo AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM
echo Quick Start for Windows
echo ================================================
echo.

REM Check if SUMO is in common locations and add to PATH
if exist "D:\sumo\bin\sumo.exe" (
    echo Found SUMO at D:\sumo\bin
    set PATH=%PATH%;D:\sumo\bin
    goto run_setup
)

if exist "C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe" (
    echo Found SUMO at C:\Program Files (x86)\Eclipse\Sumo\bin
    set PATH=%PATH%;C:\Program Files (x86)\Eclipse\Sumo\bin
    goto run_setup
)

if exist "C:\Program Files\Eclipse\Sumo\bin\sumo.exe" (
    echo Found SUMO at C:\Program Files\Eclipse\Sumo\bin
    set PATH=%PATH%;C:\Program Files\Eclipse\Sumo\bin
    goto run_setup
)

echo SUMO not found in common locations!
echo Please run: python setup_windows.py
echo Or manually add your SUMO bin folder to PATH
pause
exit

:run_setup
echo Setting up the system...
echo.

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Running system test...
python test_system.py

echo.
echo Setup complete! You can now run:
echo   python run_example.py demo
echo   python src/train.py
echo   python src/evaluate.py
echo   python src/demo.py
echo.
pause