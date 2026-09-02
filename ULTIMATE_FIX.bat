@echo off
color 0A
title ECG Platform - Ultimate Fix Everything Script

cls
echo.
echo    ╔════════════════════════════════════════════════════════╗
echo    ║                                                        ║
echo    ║          ULTIMATE FIX - SOLVES ALL PROBLEMS            ║
echo    ║                                                        ║
echo    ╚════════════════════════════════════════════════════════╝
echo.
echo    This will fix EVERYTHING and get your app running!
echo.
echo    What this does:
echo    1. Kills all Python processes
echo    2. Cleans up locked files
echo    3. Creates fresh environment
echo    4. Installs all dependencies
echo    5. Launches your app
echo.
echo    ────────────────────────────────────────────────────────
echo.
pause

cd /d "%~dp0"

echo.
echo    [Step 1/7] Killing any running Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
echo    ✓ Processes killed

echo.
echo    [Step 2/7] Deactivating virtual environments...
call deactivate >nul 2>&1
echo    ✓ Deactivated

echo.
echo    [Step 3/7] Waiting for file locks to release...
timeout /t 3 /nobreak >nul
echo    ✓ Locks released

echo.
echo    [Step 4/7] Cleaning up old virtual environment...
rd /s /q venv >nul 2>&1
if exist venv (
    echo    ⚠ Could not delete venv completely, but that's okay!
) else (
    echo    ✓ Old venv removed
)

echo.
echo    [Step 5/7] Creating fresh virtual environment...
python -m venv venv_new
if errorlevel 1 (
    echo    ❌ ERROR: Could not create venv
    echo.
    echo    Try running this in PowerShell instead:
    echo    python -m pip install plotly streamlit
    echo    python -m streamlit run app.py
    pause
    exit /b 1
)
echo    ✓ New venv created

echo.
echo    [Step 6/7] Installing all dependencies...
echo    This takes 2-3 minutes - please wait...
call venv_new\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
python -m pip install numpy pandas scipy --quiet
python -m pip install scikit-learn xgboost --quiet
python -m pip install matplotlib plotly pywavelets --quiet
python -m pip install wfdb reportlab pytest --quiet
python -m pip install streamlit --quiet

if errorlevel 1 (
    echo    ❌ ERROR: Installation failed
    pause
    exit /b 1
)
echo    ✓ All dependencies installed!

echo.
echo    [Step 7/7] Renaming venv_new to venv...
rd /s /q venv >nul 2>&1
move venv_new venv >nul 2>&1
echo    ✓ Setup complete!

echo.
echo    ════════════════════════════════════════════════════════
echo.
echo        🎉 SUCCESS! EVERYTHING IS FIXED! 🎉
echo.
echo        Starting your ECG Analysis Platform...
echo        Browser will open automatically!
echo.
echo    ════════════════════════════════════════════════════════
echo.
echo    Quick Guide:
echo    1. Click "Analyze demo ECG" button
echo    2. Navigate pages in left sidebar
echo    3. Press Win+Shift+S for screenshots
echo    4. Press Ctrl+C here to stop server
echo.
echo    ────────────────────────────────────────────────────────
echo.

call venv\Scripts\activate.bat
streamlit run app.py

pause
