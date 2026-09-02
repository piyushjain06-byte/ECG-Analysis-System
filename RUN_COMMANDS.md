# 🚀 Complete Terminal Commands - ECG Prototype

## Step-by-Step Commands to Run Your Project

### Step 1: Open PowerShell Terminal
```powershell
# Press Win + X, then select "Windows PowerShell" or "Terminal"
# OR press Win + R, type "powershell", press Enter
```

---

### Step 2: Navigate to Project Directory
```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"
```

---

### Step 3: Check Python Installation
```powershell
python --version
```
Expected output: `Python 3.13.5` (or similar)

---

### Step 4: Install Streamlit (REQUIRED - only once)
```powershell
pip install streamlit
```

Wait for installation to complete (~30 seconds)

---

### Step 5: Verify Models Are Trained (Optional Check)
```powershell
# Check if models exist
Test-Path ".\models\trained\random_forest.pkl"
```
Expected output: `True`

If it shows `False`, run:
```powershell
python -m src.ml_training
```

---

### Step 6: Launch the Streamlit App
```powershell
streamlit run app.py
```

**What happens next:**
- Streamlit will start the server
- Your browser will open automatically to: `http://localhost:8501`
- If browser doesn't open, manually go to: `http://localhost:8501`

---

### Step 7: Use the App (In Browser)

#### 7.1 - Home Page (Analysis Setup)
1. **Select Demo Type:**
   - Choose "Arrhythmia (N + PVC/APC/fusion)" for the most interesting results
   - OR "Normal sinus rhythm", "Bradycardia", "Tachycardia"

2. **Click:** `Analyze demo ECG` button (blue button)

3. **Wait:** 5-10 seconds for analysis to complete

#### 7.2 - Navigate Through Pages (Left Sidebar)
Click through these pages in order:

📍 **DSP Analysis** (Page 2)
- You'll see 4 filtering stages
- Raw vs Processed overlay
- FFT spectrum
- *Take screenshot here!*

📍 **ECG Features** (Page 3)
- R-peak detection
- RR intervals graph
- HRV metrics table
- *Take screenshot here!*

📍 **ML Prediction** (Page 4)
- Classification result (N/S/V/F)
- Confidence percentage
- Feature importance chart
- Clinical explanation
- *Take screenshot here!*

📍 **Model Performance** (Page 5)
- 4 models comparison
- Confusion matrix
- Accuracy metrics
- *Take screenshot here!*

📍 **History** (Page 6)
- Previous analysis log
- *Take screenshot here!*

---

### Step 8: Take Screenshots

#### Windows Screenshot Methods:

**Method 1: Snipping Tool (Recommended)**
```powershell
# Press: Win + Shift + S
# Then drag to select area
# Screenshot copies to clipboard
# Paste into Paint/Word: Ctrl + V
```

**Method 2: Full Screen Screenshot**
```powershell
# Press: Win + PrtScn
# Saves to: C:\Users\pankaj jain\Pictures\Screenshots\
```

**Method 3: Snip & Sketch**
```powershell
# Open Snipping Tool
snippingtool
```

#### What to Screenshot:
1. **Home page** - With demo selector
2. **DSP Analysis page** - Filtering stages
3. **ECG Features page** - R-peaks and RR intervals
4. **ML Prediction page** - Classification result
5. **Model Performance page** - Comparison chart
6. **History page** - Analysis log

---

### Step 9: Stop the Server (When Done)
```powershell
# In the PowerShell terminal, press:
Ctrl + C

# Then type: y
# Press: Enter
```

---

## 🎯 Complete Command Sequence (Copy-Paste Ready)

```powershell
# 1. Navigate to project
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# 2. Install Streamlit (first time only)
pip install streamlit

# 3. Run the app
streamlit run app.py
```

**That's it!** Browser opens automatically → Click "Analyze demo ECG" → Navigate pages → Take screenshots

---

## 🔄 Alternative: Create Screenshots Programmatically

If you want to regenerate the static screenshots (already done):

```powershell
# Make sure you're in project directory
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# Run screenshot generator
python generate_screenshots.py
```

This creates 10 PNG files in `screenshots/` folder (already exist, no need to run)

---

## 📸 Screenshot Checklist

Save these screenshots for your presentation:

- [ ] **Home Page** - Demo selection interface
- [ ] **DSP Pipeline** - 4-stage filtering visualization
- [ ] **Raw vs Processed** - Before/after overlay
- [ ] **FFT Spectrum** - Frequency analysis
- [ ] **R-Peak Detection** - Pan-Tompkins output
- [ ] **RR Intervals** - Heart rate variability
- [ ] **Feature Table** - Extracted features
- [ ] **ML Prediction** - Classification result with confidence
- [ ] **Model Comparison** - All 4 models side-by-side
- [ ] **Confusion Matrix** - Classification accuracy
- [ ] **Feature Importance** - Bar chart
- [ ] **History Log** - Analysis database

---

## 🆘 Troubleshooting Commands

### Issue: "streamlit not found"
```powershell
pip install streamlit
```

### Issue: Port already in use
```powershell
# Run on different port
streamlit run app.py --server.port 8502
```

### Issue: Models not found
```powershell
python -m src.ml_training
```

### Issue: Browser doesn't open
Manually navigate to: `http://localhost:8501`

### Check if Streamlit is running
```powershell
netstat -ano | findstr :8501
```

### Force stop Streamlit (if Ctrl+C doesn't work)
```powershell
# Find process ID
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force
```

---

## 💡 Pro Tips

### Tip 1: Run in Background
```powershell
# Start Streamlit
Start-Process powershell -ArgumentList "streamlit run app.py"
```

### Tip 2: Clear Browser Cache
If app looks broken, hard refresh:
- Press: `Ctrl + Shift + R` (in browser)

### Tip 3: Dark Mode Screenshots
In browser (while app is open):
- Press `F12` (Developer Tools)
- Click three dots (⋮)
- More Tools → Rendering
- Scroll down to "Emulate CSS media feature prefers-color-scheme"
- Select: `prefers-color-scheme: dark`

### Tip 4: High-Res Screenshots
- Use Snipping Tool with "Delay" option
- Gives you 3-10 seconds to position window perfectly

---

## 🎬 Complete Workflow (5 Minutes)

```powershell
# 1. Open PowerShell (Win + X → Windows PowerShell)

# 2. Navigate to project
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"

# 3. Install Streamlit (if not installed)
pip install streamlit

# 4. Launch app
streamlit run app.py

# 5. In browser (opens automatically):
#    - Click "Analyze demo ECG"
#    - Wait 5-10 seconds
#    - Navigate through pages (sidebar)
#    - Press Win + Shift + S for each page
#    - Take screenshot of each page

# 6. Stop server (in PowerShell):
#    - Press Ctrl + C
#    - Type: y
#    - Press Enter
```

---

## 📹 Recording a Demo Video (Optional)

### Windows Game Bar (Built-in)
```powershell
# Press: Win + G
# Click record button (or Win + Alt + R)
# Shows red recording indicator
# Stop: Win + Alt + R
# Saved to: C:\Users\pankaj jain\Videos\Captures\
```

### OBS Studio (Professional)
1. Download: https://obsproject.com/
2. Add "Window Capture" source
3. Select browser window
4. Press "Start Recording"

---

## ✅ Success Indicators

You'll know it's working when:

1. **Terminal shows:**
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. **Browser opens** automatically to the app

3. **Home page shows:**
   - Blue gradient header "Intelligent ECG Signal Analysis"
   - Demo rhythm selector dropdown
   - Blue "Analyze demo ECG" button

4. **After clicking "Analyze demo ECG":**
   - Spinner appears: "Running DSP → R-peaks → features → classification..."
   - After 5-10 seconds, you see quality metrics and signal preview
   - Sidebar shows 6-7 pages you can navigate to

5. **DSP Analysis page shows:**
   - Four stacked graphs (Raw, Baseline-corrected, Notch-filtered, Final)
   - Interactive Plotly charts (you can zoom/pan)

**If you see all this → Perfect! Take screenshots! 📸**

---

## 🎓 For Presentation

### Best Pages to Screenshot:
1. **DSP Analysis** - Shows technical depth
2. **ML Prediction** - Shows AI results
3. **Model Performance** - Shows comparison
4. **Feature Importance** - Shows explainability

### What to Highlight:
- "Not a black box - every stage is visible"
- "4 different ML models compared"
- "Pan-Tompkins algorithm implemented"
- "100% accuracy on test set"
- "Explainable AI with feature importance"

---

## 🚀 You're Ready!

Just run these 3 commands:
```powershell
cd "c:\Users\pankaj jain\OneDrive\ドキュメント\Pictures\Desktop\ecg"
pip install streamlit
streamlit run app.py
```

**Then click, navigate, and screenshot! Good luck! 🎉**
