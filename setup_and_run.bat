@echo off
echo ========================================
echo ECG Analysis Platform - Setup and Run
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

echo [3/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn xgboost matplotlib plotly pywavelets
python -m pip install wfdb reportlab pytest
python -m pip install streamlit
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

echo [4/4] Starting Streamlit app...
echo.
echo ========================================
echo Your browser will open automatically!
echo URL: http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
