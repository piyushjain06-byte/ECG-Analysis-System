@echo off
echo ========================================
echo ECG Analysis Platform - Quick Start
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo Virtual environment not found!
    echo Please run setup_and_run.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting Streamlit app...
echo ========================================
echo Your browser will open automatically!
echo URL: http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
