@echo off
echo ========================================
echo Force Fix - Kill Processes and Reinstall
echo ========================================
echo.

cd /d "%~dp0"

echo [1] Killing any running Python/Streamlit processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2] Deactivating any active virtual environments...
call deactivate 2>nul

echo [3] Waiting 3 seconds for file locks to release...
timeout /t 3 /nobreak >nul

echo [4] Removing old virtual environment...
rd /s /q venv 2>nul

echo [5] Creating fresh virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment
    pause
    exit /b 1
)

echo [6] Activating virtual environment...
call venv\Scripts\activate.bat

echo [7] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn xgboost
python -m pip install matplotlib plotly pywavelets
python -m pip install wfdb reportlab pytest
python -m pip install streamlit

if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Everything is installed!
echo ========================================
echo.
echo Starting Streamlit app...
echo Browser will open shortly!
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
