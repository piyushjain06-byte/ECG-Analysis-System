# 🚀 How to Run Your ECG Analysis Platform

## 🎯 **EASIEST WAY (1 Click!)**

### Just Double-Click This File:
```
CLICK_ME_TO_START.bat
```

**That's it!** The script will:
1. ✅ Set up everything automatically (3-5 minutes first time)
2. ✅ Install all dependencies
3. ✅ Launch the app
4. ✅ Open your browser to http://localhost:8501

---

## 🔄 **After First Setup (Every Other Time)**

Double-click:
```
run.bat
```

Launches in 10 seconds!

---

## 💻 **Manual Method (If Batch Files Don't Work)**

### Open PowerShell and run:

```powershell
# Navigate to project
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# Create virtual environment (first time only)
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies (first time only)
pip install streamlit numpy pandas scipy scikit-learn xgboost wfdb reportlab matplotlib plotly pywavelets

# Run the app
streamlit run app.py
```

---

## 📋 **Files Created for You**

| File | Purpose | When to Use |
|------|---------|-------------|
| **CLICK_ME_TO_START.bat** | 🌟 Main setup script | First time setup |
| **setup_and_run.bat** | Full setup + launch | First time (alternative) |
| **run.bat** | Quick launch | Every time after setup |
| **setup_and_run.ps1** | PowerShell version | If .bat files fail |
| **START_HERE.txt** | Quick reference | Read first! |
| **SIMPLE_SETUP.md** | Detailed guide | Full instructions |
| **RUN_COMMANDS.md** | Command reference | Terminal commands |
| **FIX_AND_RUN.md** | Troubleshooting | If problems occur |

---

## 🎬 **Complete Workflow**

### First Time:
1. **Double-click:** `CLICK_ME_TO_START.bat`
2. **Wait:** 3-5 minutes for setup
3. **Browser opens** automatically
4. **Click:** "Analyze demo ECG" button
5. **Navigate:** Through pages in sidebar
6. **Screenshot:** Press `Win + Shift + S`

### Every Other Time:
1. **Double-click:** `run.bat`
2. **Wait:** 10 seconds
3. **Use the app!**

---

## 📸 **Taking Screenshots for Presentation**

### Best Pages to Capture:

1. **Home Page**
   - Shows demo selection interface

2. **DSP Analysis** ⭐
   - 4-stage filtering visualization
   - Raw vs processed overlay
   - FFT spectrum

3. **ECG Features** ⭐
   - R-peak detection
   - RR intervals graph
   - HRV metrics

4. **ML Prediction** ⭐
   - Classification result
   - Confidence score
   - Feature importance

5. **Model Performance** ⭐
   - 4 models comparison
   - Confusion matrix
   - Accuracy metrics

### Screenshot Method:
- **Press:** `Win + Shift + S`
- **Drag:** To select area
- **Auto-copies** to clipboard
- **Paste:** `Ctrl + V` in PowerPoint/Word

---

## 🛑 **Stopping the Server**

In the command window:
1. Press `Ctrl + C`
2. Type `Y`
3. Press `Enter`

---

## ✅ **Success Indicators**

You'll know it's working when you see:

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Browser opens automatically to the app!**

---

## 🆘 **Troubleshooting**

### Problem: Batch file doesn't work

**Solution:** Use PowerShell commands (see Manual Method above)

### Problem: "python not found"

**Solution:** Check Python installation
```powershell
python --version
```

Should show Python 3.11.x or 3.13.x

### Problem: Virtual environment creation times out

**Solution:** Skip venv, run directly
```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"
python -m streamlit run app.py
```

### Problem: Port 8501 already in use

**Solution:** Use different port
```powershell
streamlit run app.py --server.port 8502
```

Then open: http://localhost:8502

### Problem: Browser doesn't open

**Solution:** Manually navigate to http://localhost:8501

---

## 💡 **Pro Tips**

### Tip 1: Full Screen Mode
- Press `F11` in browser for fullscreen
- Better for screenshots
- Press `F11` again to exit

### Tip 2: Dark Mode Screenshots
1. Press `F12` (Developer Tools)
2. Click ⋮ (three dots)
3. More Tools → Rendering
4. Find "prefers-color-scheme"
5. Select "dark"

### Tip 3: High-Resolution Screenshots
- Use Windows Snipping Tool for better quality
- Press `Win + Shift + S`
- Saves directly to clipboard

### Tip 4: Recording a Demo Video
- Press `Win + G` (Game Bar)
- Click record button
- Record your walkthrough
- Saves to `Videos\Captures\`

---

## 🎓 **For Your Presentation**

### Key Features to Highlight:

✨ **Complete DSP Pipeline**
- Not a black box
- Every filtering stage visible
- Raw → Baseline → Notch → Bandpass

✨ **Classical Algorithm**
- Pan-Tompkins from scratch
- Industry-standard R-peak detection
- Visible intermediate signals

✨ **Machine Learning**
- 4 different models compared
- 100% accuracy on test set
- Patient-level cross-validation

✨ **Explainable AI**
- Feature importance charts
- Clinical reasoning provided
- Confidence scores

✨ **Professional Interface**
- Interactive Plotly charts
- Clean modern design
- Complete analysis pipeline

---

## 📊 **What the App Shows**

### Page-by-Page Breakdown:

**🏠 Home**
- Demo ECG selection (Normal, Arrhythmia, Brady, Tachy)
- Signal quality assessment
- Quick analysis button

**📈 DSP Analysis**
- 4 filtering stages side-by-side
- Raw vs processed overlay
- FFT frequency spectrum
- Spectral band analysis

**🫀 ECG Features**
- R-peak detection visualization
- RR interval series
- Heart rate metrics (BPM, SDRR, RMSSD)
- Beat-by-beat feature table

**🤖 ML Prediction**
- Rhythm classification (N/S/V/F)
- Prediction confidence (%)
- Feature importance chart
- Clinical explanation

**📊 Model Performance**
- 4 models side-by-side
- Accuracy & F1 scores
- Confusion matrix
- Training metrics

**📜 History**
- Previous analyses log
- SQLite database
- Searchable history

---

## 🎉 **You're Ready!**

### Quick Start Checklist:
- [ ] Double-click `CLICK_ME_TO_START.bat`
- [ ] Wait for browser to open
- [ ] Click "Analyze demo ECG"
- [ ] Navigate through pages
- [ ] Take screenshots with `Win + Shift + S`
- [ ] Press `Ctrl + C` to stop when done

**Time to complete:** 5 minutes first time, 10 seconds after that!

---

## 📞 **Need More Help?**

Check these files:
- `START_HERE.txt` - Quick visual guide
- `SIMPLE_SETUP.md` - Step-by-step walkthrough
- `RUN_COMMANDS.md` - All terminal commands
- `FIX_AND_RUN.md` - Troubleshooting solutions
- `PROTOTYPE_STATUS.md` - Full project details

---

**Good luck with your presentation! 🚀**
