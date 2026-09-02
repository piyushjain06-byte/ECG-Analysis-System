@echo off
echo ========================================
echo Fixing Virtual Environment
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv" (
    echo Virtual environment not found!
    echo Please run CLICK_ME_TO_START.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing missing dependencies...
echo.
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn xgboost matplotlib plotly pywavelets
python -m pip install wfdb reportlab pytest
python -m pip install streamlit

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ All dependencies installed!
echo ========================================
echo.
echo Now run: run.bat
echo.
pause
