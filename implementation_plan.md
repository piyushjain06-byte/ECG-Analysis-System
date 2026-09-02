# Implementation Plan - Intelligent ECG Signal Analysis Platform

This system is an academic-oriented intelligent ECG signal analysis platform that demonstrates Digital Signal Processing (DSP) techniques, feature extraction, and machine learning-based cardiac rhythm classification on the MIT-BIH Arrhythmia Database.

## User Review Required

> [!IMPORTANT]
> The system requires internet access to download MIT-BIH Arrhythmia records via the `wfdb` library, but a local **synthetic / simulated ECG generator** will be built as a fallback to ensure the application works fully offline.

> [!WARNING]
> Python 3.13 is installed. Some pre-compiled binary packages (like certain older packages or niche tools) may require compiling from source if not updated, but standard libraries like `numpy`, `scipy`, `pandas`, `scikit-learn`, `plotly`, `streamlit`, and `reportlab` support Python 3.13. We will install the latest versions to ensure compatibility.

### Proposed Rhythm Classification Scheme
We propose a 4-class classification system mapping common AAMI heart beat categories from the MIT-BIH database:
1. **Normal Beat (N)** (Normal sinus rhythm or bundle branch blocks mapped together, or Normal only)
2. **Supraventricular Ectopic Beat (S)** (APC, etc.)
3. **Ventricular Ectopic Beat (V)** (PVC, etc.)
4. **Fusion Beat (F)** 

Alternatively, a simplified 3-class system:
- **Normal (N)**
- **Ventricular Ectopic (PVC / V)**
- **Other/Atrial Anomalies (S)**

For binary classification:
- **Normal (N)** vs **Arrhythmia (Abnormal)**

To make the ML models robust and interesting, we will train classical classifiers (Logistic Regression, SVM, Random Forest, and XGBoost) on features extracted from single-beat segments, using patient-level cross-validation to prevent data leakage (unseen patients in the test set).

---

## Proposed Changes

### Component 1: Environment & Requirements
We'll configure the project using a layout separating data, models, source code, Streamlit UI pages, unit tests, and generated reports.

#### [NEW] [requirements.txt](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/requirements.txt)
Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `wfdb`, `streamlit`, `reportlab`, `plotly`, `pywavelets`

#### [NEW] [README.md](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/README.md)
Detailed setup instructions, project positioning, architectural details, and run instructions.

---

### Component 2: Core DSP & Machine Learning Code (src/)

#### [NEW] [data_loader.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/data_loader.py)
- Features: Loading MIT-BIH WFDB files (using `.hea`, `.dat`, `.atr`) and CSV files.
- Fallback: A synthetic ECG generator simulating normal rhythms, bradycardia, tachycardia, premature ventricular contraction (PVC), and noise components (power-line frequency, baseline wander, high frequency) for demonstration.

#### [NEW] [signal_quality.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/signal_quality.py)
- Evaluates raw signals for missing values, saturation (flat lines), high-frequency noise ratio, and baseline wander magnitude. Calculates a quantitative signal quality index (SQI) and maps it to `GOOD`, `ACCEPTABLE`, or `POOR`.

#### [NEW] [preprocessing.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/preprocessing.py)
- Baseline wander removal (using dual-median filtering or High-Pass Butterworth).
- Notch filtering (configurable 50 Hz/60 Hz) to eliminate power-line interference.
- Band-pass filtering (0.5 Hz - 40 Hz Butterworth) for QRS clarity.
- Min-Max/Standard normalization tools suited for morphological analysis and machine learning.

#### [NEW] [signal_analysis.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/signal_analysis.py)
- **R-Peak Detection**: Implementing the Pan-Tompkins algorithm (bandpass filter, derivative, squaring, moving window integration, adaptive thresholds).
- **RR Intervals & Heart Rate**: Calculates instantaneous and average heart rates, RR standard deviation (SDRR), RMSSD.
- **FFT & Frequency Spectrum**: Computes FFT, power spectral density (PSD), and identifies dominant frequency components.
- **Wavelet Transform (CWT/DWT)**: Optional processing for multi-scale analysis.

#### [NEW] [feature_extraction.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/feature_extraction.py)
Extracts beat-by-beat feature vectors:
1. *Temporal features*: RR interval, prep-RR, post-RR, QRS width.
2. *Statistical features*: skewness, kurtosis, variance, signal energy.
3. *Morphological features*: QRS height, amplitude differences.
4. *Spectral features*: spectral energy ratios, frequency peak parameters.

#### [NEW] [ml_training.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/ml_training.py)
- Orchestrates training on extracted features.
- Model implementations: Logistic Regression, SVM, Random Forest, XGBoost.
- Handles class imbalance (weights / SMOTE / balanced splitting).
- Implements patient-level split (prevention of intra-patient leakage).
- Saves trained models to `models/trained/`.

#### [NEW] [prediction.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/prediction.py)
- Preprocesses raw upload data, performs peak detection, extracts features, and runs inference.
- Calibrates confidence probabilities.

#### [NEW] [explainability.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/explainability.py)
- Calculates feature importances for Random Forest & XGBoost.
- Provides a rule-based expert explanation (clinical metrics mapping class decisions, e.g., PVCs lead to short RR-pre and wider QRS).

#### [NEW] [report_generator.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/report_generator.py)
- Generates a PDF document with patient/record metadata, DSP before/after waveforms, peak overlay plot, power spectrum, prediction outcome, and model confidence explanation. Contains the medical disclaimer.

---

### Component 3: Database Storage (SQLite)

#### [NEW] [history_db.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/src/history_db.py)
- Sets up an SQLite database `data/history.db` tracking:
  - Record history: timestamps, heart rate, signal quality, classification class, model, and report path.
  - Feature distributions for analytics.

---

### Component 4: Streamlit Dashboard UI (app.py & pages/)

#### [NEW] [app.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/app.py)
- Main landing page & shared state manager. Handles record upload (CSV, WFDB format), demo file management, and quality assessment.

#### [NEW] [home.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/pages/1_Home.py)
- Project intro, flowchart, academic disclaimer in a sleek card view.

#### [NEW] [dsp.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/pages/2_DSP_Analysis.py)
- Controls for filtering: toggle notch filter (50/60Hz), toggle baseline correction, customize bandpass frequencies. Plot interactive charts demonstrating signal transformation stages and the FFT spectrum.

#### [NEW] [features.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/pages/3_ECG_Features.py)
- Displays detected peaks, RR intervals, HRV statistics, and the tabular extraction vector for each beat.

#### [NEW] [prediction.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/pages/4_ML_Prediction.py)
- Renders rhythm classification, classification confidence, explainable AI contribution graphs, and has a button to download the PDF report.

#### [NEW] [performance.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/pages/5_Model_Performance.py)
- Compares Logistic Regression, SVM, Random Forest, and XGBoost using interactive comparison metrics, confusion matrices, and ROC curves.

#### [NEW] [history.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/pages/6_History.py)
- Displays a searchable, filterable logs table of previous ECG analyses with option to review details.

---

### Component 5: Tests & Notebooks

#### [NEW] [test_dsp.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/tests/test_dsp.py)
Tests loading, notch filter, band-pass filter, and Pan-Tompkins peak detector.

#### [NEW] [test_ml.py](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/tests/test_ml.py)
Tests feature extraction correctness, data split logic, and model training.

#### [NEW] [01_dsp_demo.ipynb](file:///c:/Users/pankaj jain/OneDrive/ドキュメント/Pictures/Desktop/ecg/notebooks/01_dsp_demo.ipynb)
Jupyter notebook demonstrating offline analysis step-by-step.

---

## Open Questions

> [!NOTE]
> 1. **Classification Scheme**: Do you prefer 2 classes (Normal vs Arrhythmia) or a multi-class configuration? (We propose **Normal (N)**, **Supraventricular Ectopic (S)**, **Ventricular Ectopic (V)**, **Fusion Beat (F)** as it offers excellent academic presentation value).
> 2. **Pre-downloaded data**: Let's pre-download a single MIT-BIH sample (e.g. record 100) or generate it automatically during build. Do you want the system to download records dynamically when run?

---

## Verification Plan

### Automated Tests
1. Run pytest suite:
   ```bash
   pytest tests/
   ```
2. Verify models train successfully without overfitting parameters.

### Manual Verification
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload demo data/synthetic data, adjust notch filter frequencies (50Hz vs 60Hz), baseline correction thresholds, and verify graphs render correctly.
3. Validate classification response and confirm PDF reports render matching signals and statistics.
