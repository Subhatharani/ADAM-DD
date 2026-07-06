@echo off
REM ADAM-DD: Quick Start Script for Windows
REM This script sets up the environment and runs the application

echo.
echo ====================================
echo ADAM-DD: Quick Start Setup
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/5] Python found. Creating virtual environment...
python -m venv adam_dd_env

echo [2/5] Activating virtual environment...
call adam_dd_env\Scripts\activate.bat

echo [3/5] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [4/5] Creating templates folder...
if not exist templates mkdir templates

echo [5/5] Done! Starting ADAM-DD...
echo.
echo ====================================
echo Starting Flask Server...
echo ====================================
echo.
echo Open your browser to: http://localhost:5000
echo.
python main.py

pause
