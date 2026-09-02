@echo off
color 0A
title ECG Analysis Platform - Easy Setup

echo.
echo    ╔════════════════════════════════════════════════════════╗
echo    ║                                                        ║
echo    ║        ECG ANALYSIS PLATFORM - AUTO SETUP              ║
echo    ║                                                        ║
echo    ╚════════════════════════════════════════════════════════╝
echo.
echo.
echo    This script will:
echo    ✓ Create virtual environment
echo    ✓ Install all dependencies
echo    ✓ Launch the web app
echo    ✓ Open your browser automatically
echo.
echo    Time needed: 3-5 minutes (first time only)
echo.
echo    ────────────────────────────────────────────────────────
echo.
pause

cd /d "%~dp0"

echo.
echo    [Step 1/4] Creating virtual environment...
echo    Please wait...
echo.
python -m venv venv
if errorlevel 1 goto error

echo    ✓ Virtual environment created!
echo.
echo    [Step 2/4] Activating virtual environment...
echo.
call venv\Scripts\activate.bat
if errorlevel 1 goto error

echo    ✓ Virtual environment activated!
echo.
echo    [Step 3/4] Installing dependencies...
echo    This may take 2-3 minutes...
echo.
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn xgboost matplotlib plotly pywavelets
python -m pip install wfdb reportlab pytest
python -m pip install streamlit
if errorlevel 1 goto error

echo.
echo    ✓ Dependencies installed successfully!
echo.
echo    [Step 4/4] Launching Streamlit app...
echo.
echo    ════════════════════════════════════════════════════════
echo.
echo        🎉 SUCCESS! Your browser will open shortly! 🎉
echo.
echo        URL: http://localhost:8501
echo.
echo    ════════════════════════════════════════════════════════
echo.
echo    INSTRUCTIONS:
echo    1. Browser opens automatically
echo    2. Click "Analyze demo ECG" button
echo    3. Navigate pages in left sidebar
echo    4. Press Win+Shift+S for screenshots
echo.
echo    To stop: Press Ctrl+C in this window
echo.
echo    ────────────────────────────────────────────────────────
echo.

streamlit run app.py
goto end

:error
echo.
echo    ════════════════════════════════════════════════════════
echo    ❌ ERROR: Setup failed!
echo    ════════════════════════════════════════════════════════
echo.
echo    Try these manual commands in PowerShell:
echo.
echo    cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"
echo    python -m venv venv
echo    .\venv\Scripts\Activate.ps1
echo    pip install streamlit
echo    streamlit run app.py
echo.
pause
exit /b 1

:end
pause
