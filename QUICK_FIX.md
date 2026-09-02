# 🔧 Quick Fix for "ModuleNotFoundError: No module named 'plotly'"

## ✅ Solution 1: Fix Your Existing Virtual Environment (FASTEST)

Double-click this file:
```
fix_venv.bat
```

This will install all missing dependencies in your current venv.

Then run:
```
run.bat
```

---

## ✅ Solution 2: Fresh Virtual Environment

If the above doesn't work, delete the old venv and start fresh:

### Step 1: Delete Old Virtual Environment
```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"
Remove-Item -Recurse -Force venv
```

### Step 2: Run Setup Again
Double-click:
```
CLICK_ME_TO_START.bat
```

---

## ✅ Solution 3: Manual Fix (PowerShell)

Open PowerShell and run:

```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# Activate your existing venv
.\venv\Scripts\Activate.ps1

# Install all dependencies one by one
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install pandas
python -m pip install scipy
python -m pip install scikit-learn
python -m pip install xgboost
python -m pip install matplotlib
python -m pip install plotly
python -m pip install pywavelets
python -m pip install wfdb
python -m pip install reportlab
python -m pip install pytest
python -m pip install streamlit

# Now run the app
streamlit run app.py
```

---

## ✅ Solution 4: Skip Virtual Environment (Quick Test)

If you just want to test quickly:

```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# Install globally (not recommended but works)
pip install plotly

# Run with system Python
python -m streamlit run app.py
```

---

## 🎯 What Caused This Error?

The error `ModuleNotFoundError: No module named 'plotly'` means:
- Plotly wasn't installed in your virtual environment
- The batch script might have failed silently
- Some dependencies weren't properly installed

---

## 📋 Complete Dependency List

Your app needs these packages:
- ✅ numpy
- ✅ pandas
- ✅ scipy
- ✅ scikit-learn
- ✅ xgboost
- ✅ matplotlib
- ❌ **plotly** (this was missing!)
- ✅ pywavelets
- ✅ wfdb
- ✅ reportlab
- ✅ pytest
- ✅ streamlit

---

## 🔍 Verify Installation

After fixing, verify plotly is installed:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import plotly; print(plotly.__version__)"
```

Should show: `7.0.0` or similar (not an error)

---

## ✅ Recommended Steps (In Order)

1. **First try:** Double-click `fix_venv.bat`
2. **Then run:** Double-click `run.bat`
3. **If that fails:** Use Solution 2 (fresh venv)
4. **If still fails:** Use Solution 3 (manual install)

---

## 🆘 Still Not Working?

Try this nuclear option:

```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# Delete venv completely
Remove-Item -Recurse -Force venv

# Create new venv
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install requirements from file
pip install -r requirements.txt

# Run app
streamlit run app.py
```

This uses the `requirements.txt` file which has all dependencies listed.

---

## 💡 Prevention

To avoid this in the future:
- Always activate venv before running app
- Use `requirements.txt` for consistent installs
- Check for errors during pip install

---

**TL;DR:** Double-click `fix_venv.bat` then `run.bat` 🚀
