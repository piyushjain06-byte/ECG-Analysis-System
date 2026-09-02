# ECG Platform - Simple Run Script (No venv needed!)
Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   ECG Platform - Simple Run Mode       ║" -ForegroundColor Cyan  
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

Write-Host "Installing missing packages..." -ForegroundColor Yellow
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas scipy scikit-learn xgboost matplotlib plotly pywavelets wfdb reportlab streamlit

Write-Host ""
Write-Host "✓ Packages installed!" -ForegroundColor Green
Write-Host ""
Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Starting app..." -ForegroundColor Green
Write-Host "  Browser opens at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

python -m streamlit run app.py
