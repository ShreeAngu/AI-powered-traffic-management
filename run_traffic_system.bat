@echo off
title AI Traffic Light System
echo ================================================
echo AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM
echo ================================================
echo.

REM Add SUMO to PATH for this session
set PATH=%PATH%;D:\sumo\bin

echo SUMO Path: D:\sumo\bin
echo Python: D:\Python\python.exe
echo.

:menu
echo Available Options:
echo 1. Quick Demo
echo 2. Train Models
echo 3. Evaluate Models  
echo 4. Interactive Demo
echo 5. System Test
echo 6. Exit
echo.
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto train
if "%choice%"=="3" goto evaluate
if "%choice%"=="4" goto interactive
if "%choice%"=="5" goto test
if "%choice%"=="6" goto exit
goto menu

:demo
echo Running quick demo...
python run_example.py demo
pause
goto menu

:train
echo Starting training...
python src/train.py
pause
goto menu

:evaluate
echo Running evaluation...
python src/evaluate.py
pause
goto menu

:interactive
echo Starting interactive demo...
python src/demo.py
pause
goto menu

:test
echo Running system test...
python test_system.py
pause
goto menu

:exit
echo Goodbye!
pause
