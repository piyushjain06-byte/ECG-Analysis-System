# **IMPLEMENTATION PLAN V2** 

# **Intelligent ECG Signal Analysis Platform** 

**Version:** 2.0 

**Project Type:** Academic / Educational ECG Analysis System **Primary Framework:** Python + Streamlit **Dataset:** MIT-BIH Arrhythmia Database **ML:** Logistic Regression, SVM, Random Forest, XGBoost **DSP:** SciPy + PyWavelets **Database:** SQLite 

**Reports:** ReportLab PDF 

# **1. PROJECT OBJECTIVE** 

Build an **AI-powered ECG Signal Analysis Platform** that demonstrates the complete ECG analysis pipeline: 

ECG Input 

↓ 

Data Validation 

↓ 

Signal Quality Assessment 

↓ 

Preprocessing / Noise Removal 

↓ 

R-Peak Detection 

↓ 

Heart Rate & RR Analysis 

↓ 

ECG Feature Extraction 

↓ 

Machine Learning Classification 

↓ 

Prediction + Confidence 

↓ 

Explainable AI 

↓ 

Visualization 

↓ 

History Storage 

↓ 

PDF Report 

The system is intended for **academic demonstration and signal-analysis research only** . 

It must **not claim to diagnose medical conditions or replace a qualified medical professional** . 

# **2. FINAL TECHNOLOGY STACK** 

# **Programming** 

- Python 3.13 

- Object-oriented / modular Python architecture 

# **UI** 

- Streamlit 

- Plotly 

# **Numerical / DSP** 

- NumPy 

- SciPy 

- PyWavelets 

# **Data** 

- Pandas 

- WFDB 

# **Machine Learning** 

- Scikit-learn 

- XGBoost 

# **Database** 

- SQLite 

- Python sqlite3 

# **Reports** 

- ReportLab 

# **Testing** 

- Pytest 

# **3. FINAL CLASSIFICATION SCHEME** 

# The project will use **4-class ECG beat classification** . 

This decision replaces the open-ended classification choice in Version 1, which proposed 4-class, 3- class and binary alternatives. 

# **Classes** 

# **Class Meaning** 

- N Normal 

- S Supraventricular ectopic 

- V Ventricular ectopic 

- F Fusion 

# **Proposed AAMI mapping** 

N class: 

N, L, R, e, j 

S class: 

- A, a, J, S 

V class: 

V, E 

F class: 

F 

Other annotation symbols that do not belong to these four classes will be excluded from supervised beat classification. 

**Important:** This mapping must be implemented centrally in one configuration/module so that training, evaluation and documentation all use exactly the same mapping. 

# **4. DATASET STRATEGY** 

**Primary Dataset** 

Use the: 

# **MIT-BIH Arrhythmia Database** 

through the wfdb Python library. 

The original plan already specifies support for .hea, .dat, .atr files. 

# **5. DATA LOADING** 

The system must support: 

# **Option A — WFDB** 

record.hea 

record.dat 

record.atr 

# **Option B — CSV** 

CSV support should accept common formats such as: 

time,ecg 

0.000,0.12 

0.002,0.14 

... 

or: 

ecg 

0.12 

0.14 

... 

The system should allow the sampling frequency to be specified if it is not available inside the file. 

# **6. SYNTHETIC ECG FALLBACK** 

The original plan includes a synthetic ECG generator so that the application can operate without downloading MIT-BIH data. 

Version 2 defines this more strictly. 

# **Synthetic signals** 

Generate: 

- Normal ECG 

- Bradycardia 

- Tachycardia 

- PVC-like beats 

- Baseline wander 

- Power-line noise 

- High-frequency noise 

# **Default configuration** 

Sampling frequency: 360 Hz 

Duration: configurable 

Default duration: 10 seconds 

Synthetic data must be used for: 

- demonstration 

- testing 

- UI development 

- offline operation 

# **Important** 

Synthetic ECG must **not automatically be mixed with MIT-BIH data during final model training** . 

The final reported ML performance should be based on the real MIT-BIH dataset. 

# **7. DATA VALIDATION** 

Every uploaded ECG must go through validation. 

Check: 

- File format 

- Number of samples 

- Sampling frequency 

- Missing values 

- NaN 

- Infinite values 

- Flat-line signal 

- Extremely large amplitudes 

- Insufficient signal duration 

- Invalid WFDB files 

- Missing annotations where classification is required 

If invalid: 

ECG file could not be analyzed. 

Reason: 

<clear error message> 

Suggested action: 

<what the user should do> 

The application must never crash because of a bad upload. 

# **8. SIGNAL QUALITY ASSESSMENT** 

Create: 

src/signal_quality.py 

The original plan already defines an SQI system based on missing values, saturation, high-frequency noise and baseline wander. 

# **Metrics** 

Calculate: 

- Missing-value ratio 

- Flat-line percentage 

- Baseline wander magnitude 

- High-frequency noise ratio 

- Signal amplitude range 

- Signal-to-noise related indicators 

# **Final quality score** 

Create: 

Signal Quality Index (SQI) 

Output: 

GOOD 

ACCEPTABLE 

POOR 

Example: 

SQI: 87/100 

Status: GOOD 

If signal quality is extremely poor, the system should warn the user before prediction. 

# **9. PREPROCESSING PIPELINE** 

Create: 

src/preprocessing.py 

The original plan specifies baseline correction, notch filtering and band-pass filtering. 

# **Stage 1 — Baseline Wander Removal** 

Use configurable: 

- High-pass Butterworth 

- Optional median filtering approach 

Default: 

High-pass cutoff = 0.5 Hz 

# **Stage 2 — Power-Line Noise Removal** 

Configurable: 

50 Hz 

60 Hz 

Use a notch filter. 

Default: 

50 Hz 

because this is appropriate for the primary India-oriented demonstration environment. 

# **Stage 3 — Band-Pass Filtering** 

Default: 

0.5 Hz – 40 Hz 

Butterworth filter. 

# **Stage 4 — Normalization** 

Support: 

- StandardScaler-style normalization for ML 

- Min-Max normalization where appropriate 

**Important:** The ML scaler must be fitted only on training data and reused during inference. 

# **10. DSP ANALYSIS** 

Create: 

src/signal_analysis.py 

The original plan includes Pan-Tompkins, RR analysis, FFT/PSD and wavelets. 

# **11. R-PEAK DETECTION** 

Implement a Pan-Tompkins-style algorithm. 

Pipeline: 

Filtered ECG 

↓ 

Derivative 

↓ 

Squaring 

↓ 

Moving Window Integration 

↓ 

Adaptive Threshold 

↓ 

Candidate Peaks 

↓ 

Refractory-period filtering 

↓ 

R-peaks 

Return: 

peak_indices 

peak_times 

peak_amplitudes 

Also include sanity checks to prevent impossible heart rates. 

# **12. HEART RATE ANALYSIS** 

Calculate: 

# **Instantaneous HR** 

HR = 60 / RR 

# **Average HR** 

Calculate over detected beats. 

# **RR intervals** 

Store: 

RR 

RR-pre 

RR-post 

# **HRV-style metrics** 

Calculate: 

- Mean RR 

- SDRR / SDNN-style metric 

- RMSSD 

The original plan explicitly proposed RR intervals, heart rate, SDRR and RMSSD. 

# **13. FFT / FREQUENCY ANALYSIS** 

Calculate: 

- FFT 

- Frequency spectrum 

- Power spectral density 

- Dominant frequencies 

- Spectral energy 

Display: 

Time Domain 

Frequency Domain 

# **14. WAVELET ANALYSIS** 

Implement optional: 

- DWT 

- CWT 

Use wavelets for: 

- multi-scale ECG analysis 

- optional feature generation 

- academic DSP demonstration 

Wavelet processing should not make the primary pipeline unnecessarily slow. 

# **15. FEATURE EXTRACTION** 

Create: 

src/feature_extraction.py 

The original plan already divides features into temporal, statistical, morphological and spectral groups. 

Version 2 defines the feature groups more explicitly. 

# **A. Temporal Features** 

Per beat: 

RR interval 

RR-pre 

RR-post 

RR ratio 

# **B. Morphological Features** 

Extract: 

QRS amplitude 

QRS width 

Beat amplitude 

Maximum amplitude 

Minimum amplitude 

Peak-to-peak amplitude 

Where feasible: 

R-peak amplitude 

QRS area 

# **C. Statistical Features** 

Calculate: 

Mean 

Standard deviation 

Variance 

Skewness 

Kurtosis 

Energy 

RMS 

# **D. Spectral Features** 

Calculate: 

Dominant frequency 

Spectral energy 

Low-frequency energy 

High-frequency energy 

Energy ratios 

# **16. FEATURE VECTOR** 

Every beat must eventually become a fixed-length numerical vector. 

Example: 

[ 

RR_pre, 

RR_post, 

RR, 

QRS_width, 

R_amplitude, 

QRS_area, 

mean, 

std, 

variance, 

skewness, 

kurtosis, 

energy, 

dominant_frequency, 

spectral_energy, 

... 

] 

The exact feature order must be fixed and stored. 

Create metadata describing: 

feature_name 

feature_index 

feature_description 

This is important for model reproducibility and explainability. 

# **17. DATA SPLITTING — CRITICAL** 

This is one of the most important requirements. 

**Never randomly split individual beats from the same patient into training and testing.** 

The original plan already correctly requires patient-level splitting to prevent leakage. 

**Required approach** 

Split at **record/patient level first** . 

Recommended: 

70% → Training 

15% → Validation 

15% → Test 

The exact record assignment must be saved so experiments can be reproduced. 

# **18. DATA LEAKAGE PREVENTION** 

The following must happen **only after the patient-level split** : 

- Scaling 

- Feature selection 

- SMOTE 

- Any learned preprocessing 

- Hyperparameter tuning 

Never fit a scaler on the complete dataset before splitting. 

Never use test data during training. 

Never use test data for hyperparameter selection. 

# **19. CLASS IMBALANCE** 

MIT-BIH classes are not naturally balanced. 

Possible techniques: 

**Primary approach** 

Use: 

class_weight="balanced" 

where supported. 

**Optional** 

SMOTE may be used on the **training set only** . 

Never apply SMOTE to validation or test data. 

The original plan already proposed weights / SMOTE / balanced splitting. 

# **20. MACHINE LEARNING MODELS** 

Train four models: 

**Model 1** 

Logistic Regression 

Purpose: 

- baseline 

- interpretable 

- fast 

# **Model 2** 

Support Vector Machine 

Purpose: 

- nonlinear/classification comparison 

- strong classical ML benchmark 

# **Model 3** 

Random Forest 

Purpose: 

- feature importance 

- robust baseline 

**Model 4** 

XGBoost 

Purpose: 

- high-performance classical boosting model 

These four models were already specified in Version 1. 

# **21. HYPERPARAMETER TUNING** 

Do not blindly use default parameters. 

Use either: 

GridSearchCV 

or: 

RandomizedSearchCV 

with patient-aware cross-validation. 

Tune reasonable parameters such as: 

# **Logistic Regression** 

C 

solver 

# **SVM** 

C 

kernel 

gamma 

# **Random Forest** 

n_estimators 

max_depth 

min_samples_split 

**XGBoost** 

n_estimators 

max_depth learning_rate subsample 

Do not perform excessive tuning that causes the project to become unnecessarily slow. 

# **22. MODEL EVALUATION** 

The final evaluation must include: 

# **Basic metrics** 

- Accuracy 

- Precision 

- Recall 

- F1-score 

# **Important imbalanced-data metrics** 

- Macro F1 

- Weighted F1 

- Per-class precision 

- Per-class recall 

- Per-class F1 

# **Additional** 

- Confusion matrix 

- ROC-AUC 

- PR-AUC 

- Sensitivity 

- Specificity 

# **23. MODEL COMPARISON** 

Create a comparison table: 

**Model Accuracy Macro F1 Weighted F1 ROC-AUC Training Time** 

Logistic Regression 

SVM 

Random Forest 

XGBoost 

The best model should **not automatically mean the highest accuracy** . 

Prefer the model based on overall performance, especially Macro F1 and per-class performance. 

# **24. MODEL SAVING** 

Save: 

models/trained/ 

Store: 

logistic_regression.pkl 

support_vector_machine.pkl 

random_forest.pkl 

xgboost.pkl 

scaler.pkl 

label_encoder.pkl 

Also save metadata: 

models/metadata/ml_performance.json 

Metadata should contain: 

training date 

dataset 

classification mapping 

feature list 

train/test split 

metrics 

model parameters 

best model 

Python/package versions where practical 

# **25. MODEL VERSIONING** 

Every trained model should have a version. 

Example: 

model_version: v1.0 

dataset_version: MIT-BIH 

feature_version: v1 

classification_version: 4-class-AAMI 

This prevents confusion if models are retrained later. 

# **26. PREDICTION PIPELINE** 

Create: 

src/prediction.py 

Prediction must follow: 

Uploaded ECG 

↓ 

Validation 

↓ 

Signal Quality 

↓ 

Preprocessing 

↓ 

R-Peak Detection 

↓ 

Beat Segmentation 

↓ 

Feature Extraction 

↓ 

Feature Scaling 

↓ 

ML Model 

↓ 

Prediction ↓ 

Probability 

↓ 

Explanation 

The original plan already defines this overall inference responsibility. 

# **27. CONFIDENCE** 

Display: 

Prediction: 

Ventricular Ectopic 

Confidence: 

93.4% 

Also show probability distribution: 

Normal:       2.1% Supraventricular: 1.8% Ventricular:  93.4% 

Fusion:       2.7% 

If probability is low or ambiguous, display: 

⚠ Low-confidence prediction 

Do not represent model probability as medical certainty. 

# **28. EXPLAINABLE AI** 

Create: 

src/explainability.py 

The original plan proposes feature importance and rule-based explanations. 

Implement: 

# **Global explanation** 

Which features generally influence the model. 

# **Per-prediction explanation** 

Which features were important for the current prediction. 

Example: 

Prediction: 

Ventricular Ectopic 

Important features: 

QRS Width 

RR-pre 

RR-post 

QRS Amplitude 

# **29. RULE-BASED EXPLANATION** 

Generate understandable explanations. 

Example: 

The prediction is associated with: 

- increased QRS width 

- abnormal RR interval 

- altered beat morphology 

These signal characteristics contributed to the 

machine-learning classification. 

Clearly label this as: 

Educational signal interpretation 

—not a clinical diagnosis. 

# **30. APPLICATION ARCHITECTURE** 

The application should follow this architecture: 

STREAMLIT UI 

│ ▼ Application Layer 



<!-- Start of picture text -->
                       │<br>▼<br>                 ECG Pipeline<br>                       │<br>       ┌─────────────── ┼ ───────────────┐<br>▼ ▼ ▼<br> Data Loader      Signal Quality    Preprocessing<br>       │               │               │<br>       └─────────────── ┼ ───────────────┘<br>▼<br>                Signal Analysis<br>                       │<br>▼<br>                Feature Extraction<br>                       │<br>▼<br>                 ML Prediction<br>                       │<br>┴<br>            ┌────────── ──────────┐<br>▼ ▼<br>      Explainability          Evaluation<br>            │<br>┴<br>       ┌──── ────┐<br>▼ ▼<br>    SQLite      PDF<br><!-- End of picture text -->

# **31. PROJECT FILE STRUCTURE** 

The final architecture should align with the existing project rather than creating unnecessary duplicate files. 

ECG-Analysis-System/ 

│ 

`├` ── app.py 

`├` ── implementation_plan.md 

`├` ── requirements.txt 

`├` ── README.md 

`├` ── .gitignore 

│ 

`├` ── .streamlit/ 

│   └── config.toml 

│ 

`├` ── src/ 

│ `├` ── __init__.py 

│ `├` ── data_loader.py 

│ `├` ── signal_quality.py 

│ `├` ── preprocessing.py 

- │ `├` ── signal_analysis.py 

- │ `├` ── feature_extraction.py 

- │ `├` ── ml_training.py 

- │ `├` ── prediction.py 

- │ `├` ── explainability.py 

- │ `├` ── report_generator.py 

- │ `├` ── history_db.py 

- │ `├` ── pipeline.py 

- │ `├` ── visualization.py 

- │ `├` ── ui_pages.py 

- │   └── mitbih_data.py 

│ 

- `├` ── models/ 

- │ `├` ── trained/ 

- │   └── metadata/ 

- │       └── ml_performance.json 

│ 

`├` ── data/ 

│ `├` ── sample/ 

- │   └── mitbih/ 

│ 

`├` ── tests/ 

│ `├` ── test_pipeline.py 

- │ `├` ── test_dsp.py 

- │   └── test_ml.py 

│ 

- `├` ── notebooks/ 

- │   └── 01_dsp_demo.ipynb 

│ 

`├` ── screenshots/ 

│ 

└── reports/ 

# **32. IMPORTANT EXISTING-REPOSITORY RULE** 

Claude must **inspect the existing repository before creating or deleting files** . 

Do not create duplicate implementations such as: 

pages/ 

if the current project architecture already implements those functions through: 

src/ui_pages.py 

First determine what already exists. 

Then modify the existing implementation where appropriate. 

# **33. STREAMLIT UI** 

The dashboard should provide these major sections. 

# **PAGE 1 — HOME** 

Display: 

- Project title 

- Project description 

- System workflow 

- Technology stack 

- Academic purpose 

- Disclaimer 

Example: 

Intelligent ECG Signal Analysis Platform 

DSP + Machine Learning + Explainable AI 

# **34. PAGE 2 — ECG INPUT** 

Allow: 

Upload CSV 

Upload WFDB 

Use Demo ECG 

Generate Synthetic ECG 

Show: 

Sampling Rate 

Duration 

Number of Samples 

Signal Quality 

Plot raw ECG. 

# **35. PAGE 3 — DSP ANALYSIS** 

Show: 

**Raw signal** 

Original ECG 

**Processing stages** 

Raw ↓ Baseline corrected ↓ Notch filtered ↓ Band-pass filtered Allow user controls: 

Notch: ON/OFF Frequency: 50/60 Hz Baseline correction: ON/OFF Band-pass low cutoff Band-pass high cutoff 

# **36. FFT DASHBOARD** 

Display: 

- FFT 

- PSD 

- dominant frequency 

- spectral energy 

Use interactive Plotly charts. 

# **37. PAGE 4 — ECG FEATURES** 

Display: 

R-peaks 

RR intervals 

Heart rate 

HRV metrics 

QRS-related features 

Statistical features 

Spectral features 

Show detected R-peaks overlaid on ECG. 

# **38. PAGE 5 — ML PREDICTION** 

Display: 

Selected Model 

Prediction 

Confidence 

Probability distribution 

Then: 

Explainable AI 

with: 

- top features 

- feature importance 

- textual explanation 

# **39. PAGE 6 — MODEL PERFORMANCE** 

Display: 

# **Model comparison** 

Logistic Regression 

SVM 

Random Forest 

XGBoost 

Charts: 

- Accuracy 

- Macro F1 

- Weighted F1 

- ROC-AUC 

- Confusion matrix 

- ROC curves 

- Precision/Recall comparison 

# **40. PAGE 7 — HISTORY** 

Show previous analyses. 

Columns: 

ID 

Timestamp 

Record 

Heart Rate 

Signal Quality 

Prediction 

Confidence 

Model 

Allow: 

- search 

- filtering 

- viewing analysis 

- report access 

# **41. DATABASE DESIGN** 

Create: 

data/history.db 

# **Table: analysis_history** 

id 

timestamp 

record_name 

patient_id 

sampling_rate 

duration 

heart_rate 

signal_quality 

predicted_class 

confidence 

model_name 

model_version 

report_path 

# **Table: analysis_features** 

id 

analysis_id 

feature_name 

feature_value 

# **Optional Table: model_runs** 

id 

model_name 

model_version 

dataset 

accuracy 

macro_f1 

weighted_f1 

roc_auc 

training_timestamp 

# **42. PDF REPORT** 

Create: 

src/report_generator.py 

The original plan already requires a report containing metadata, before/after DSP plots, peaks, spectrum, prediction and confidence explanation. 

PDF should contain: 

**Section 1** 

Project information 

**Section 2** 

ECG record information 

**Section 3** 

Signal quality 

**Section 4** 

Raw ECG 

**Section 5** 

Processed ECG 

**Section 6** 

R-peak detection 

**Section 7** 

Heart rate / RR analysis 

**Section 8** 

FFT / spectrum 

**Section 9** 

ML prediction 

**Section 10** 

Confidence 

**Section 11** 

Explainability 

**Section 12** 

Disclaimer 

# **43. TESTING** 

Create: 

tests/test_dsp.py 

tests/test_ml.py 

tests/test_pipeline.py 

The original plan already requires automated testing through pytest. 

# **44. DSP TESTS** 

Test: 

- ECG loading 

- sampling rate handling 

- notch filter 

- band-pass filter 

- baseline correction 

- R-peak detection 

- RR calculation 

- heart rate calculation 

- FFT 

# **45. FEATURE TESTS** 

# Test: 

- feature vector length 

- no NaN 

- no infinite values 

- expected feature names 

- deterministic output 

# **46. ML TESTS** 

Test: 

- dataset splitting 

- patient leakage 

- feature scaling 

- training 

- prediction 

- probability output 

- model saving/loading 

# **47. PIPELINE TEST** 

Create one end-to-end test: 

Sample ECG 

↓ 

Quality ↓ 

Preprocessing ↓ 

Peaks 

↓ 

Features ↓ Model ↓ 

Prediction 

The pipeline must successfully complete without crashing. 

# **48. ERROR HANDLING** 

The UI must gracefully handle: 

# **No R-peaks** 

Unable to detect reliable R-peaks. 

Please check signal quality. 

# **Too-short signal** 

Signal is too short for reliable analysis. 

# **Invalid file** 

Unsupported or corrupted ECG file. 

# **Missing sampling rate** 

Ask user for it. 

# **Model unavailable** 

Prediction model has not been trained. 

Please train/load a model first. 

# **49. OFFLINE MODE** 

The application must have two modes. 

**Online** 

Download MIT-BIH using WFDB 

# **Offline** 

Use local sample ECG 

Use synthetic ECG 

Use locally stored trained models 

The user should be able to demonstrate the application without internet. 

# **50. REPRODUCIBILITY** 

The project should use fixed random seeds where randomness is involved. 

For example: 

random_state = 42 

Store: 

- split information 

- feature list 

- model parameters 

- classification mapping 

- dataset version 

- model version 

This ensures experiments can be reproduced. 

# **51. SECURITY / PRIVACY** 

The system should: 

- avoid external API calls for ECG analysis 

- process uploaded signals locally 

- avoid storing unnecessary personal information 

- avoid real patient identifiers 

- clearly state academic purpose 

- display a medical disclaimer 

# **52. PERFORMANCE REQUIREMENTS** 

The application should feel responsive for demonstration. 

Target: 

# **Synthetic ECG** 

Near-instant generation. 

# **Sample ECG** 

A few seconds or less for normal analysis. 

# **Prediction** 

Near-instant once models are loaded. 

# **Model training** 

Can take longer and should show a progress indicator. 

Models should preferably be loaded once rather than reloaded for every Streamlit interaction. 

# **53. STREAMLIT STATE MANAGEMENT** 

Use Streamlit session state to avoid unnecessarily repeating: 

- file loading 

- preprocessing 

- peak detection 

- feature extraction 

- model loading 

Example conceptual state: 

raw_signal 

processed_signal 

sampling_rate 

r_peaks 

rr_intervals 

features 

prediction 

confidence 

selected_model 

# **54. CACHING** 

Use Streamlit caching appropriately for: 

- loading models 

- loading datasets 

- expensive preprocessing where appropriate 

Do not cache mutable state incorrectly. 

# **55. VISUALIZATION REQUIREMENTS** 

Use Plotly for interactive graphs where appropriate. 

Charts should have: 

- titles 

- axis labels 

- units 

- legends 

- readable scale 

- hover information 

ECG plots should clearly show: 

Time (seconds) 

Amplitude 

# **56. USER WORKFLOW** 

The complete user experience should be: 

1. Open application 

↓ 

2. Select ECG source 

↓ 

3. Load ECG 

↓ 

4. Validate ECG 

↓ 

5. Calculate signal quality 

↓ 

6. View raw ECG 

↓ 

7. Apply DSP ↓ 

8. Detect R-peaks 

↓ 

9. Calculate HR/RR ↓ 

10. Extract features ↓ 

11. Select ML model ↓ 

12. Generate prediction ↓ 

13. Show confidence 

↓ 

14. Show explanation 

↓ 

15. Save history 

↓ 

16. Generate PDF 

# **57. IMPLEMENTATION ORDER** 

Claude should implement in this order. 

# **Phase 1 — Foundation** 

- Verify environment 

- Verify requirements 

- Verify project structure 

- Verify current code 

- Fix imports 

- Fix configuration 

# **Phase 2 — Data Layer** 

Implement/fix: 

data_loader.py 

mitbih_data.py 

Verify: 

- CSV 

- WFDB 

- synthetic ECG 

# **Phase 3 — Signal Quality** 

Implement: 

signal_quality.py 

Test quality scoring. 

# **Phase 4 — DSP** 

Implement/fix: preprocessing.py 

signal_analysis.py 

Verify: 

- filters 

- R-peaks 

- RR 

- HR 

- FFT 

- wavelets 

# **Phase 5 — Features** 

Implement: 

feature_extraction.py 

Verify fixed feature vectors. 

# **Phase 6 — ML** 

Implement: 

ml_training.py 

Complete: 

dataset 

→ split 

→ scaling 

→ imbalance 

→ training 

→ evaluation 

→ model saving 

# **Phase 7 — Prediction** 

Implement: 

prediction.py 

Connect trained models to inference. 

# **Phase 8 — Explainability** 

Implement: 

explainability.py 

# **Phase 9 — Database** 

Implement: 

history_db.py 

# **Phase 10 — Visualization** 

Implement/fix: 

visualization.py 

# **Phase 11 — Streamlit** 

Implement/fix: 

app.py 

ui_pages.py 

Connect the entire pipeline. 

# **Phase 12 — PDF** 

Implement: 

report_generator.py 

# **Phase 13 — Testing** 

Run: 

pytest tests/ 

# **Phase 14 — Final UI Polish** 

Improve: 

- layout 

- cards 

- metrics 

- charts 

- warnings 

- navigation 

- loading indicators 

- error messages 

# **58. ACCEPTANCE CRITERIA** 

The project is considered complete only when all of the following work. 

# **Data** 

- CSV upload works 

- WFDB loading works 

- Synthetic ECG works 

- MIT-BIH sample works 

# **DSP** 

- Raw ECG visualization 

- Baseline correction 

- Notch filtering 

- Band-pass filtering 

- R-peak detection 

- RR intervals 

- Heart rate 

- FFT 

- PSD 

- Optional wavelets 

# **ML** 

- 4-class classification 

- Patient-level split 

- No data leakage 

- Class imbalance handling 

- • Logistic Regression 

- SVM 

- Random Forest 

- XGBoost 

- Evaluation metrics 

- Confusion matrix 

- ROC curves 

- Model comparison 

# **Explainability** 

- Feature importance 

- Prediction explanation 

- Confidence/probabilities 

# **Database** 

- Analysis history 

- Feature storage 

- Model information 

# **Reports** 

- PDF generation 

- ECG plots 

- DSP plots 

- prediction 

- confidence 

- explanation 

- disclaimer 

# **UI** 

- Home 

- Upload 

- DSP 

- Features 

- Prediction 

- Performance 

- History 

# **Testing** 

- Unit tests pass 

- Pipeline test passes 

- Application launches 

- No critical exceptions 

# **59. FINAL VALIDATION** 

Run: 

pytest tests/ 

Then: 

streamlit run app.py 

Perform a complete manual test: 

Synthetic ECG 

↓ DSP ↓ Features ↓ Prediction 

↓ Explanation 

↓ History ↓ PDF 

Then test with an MIT-BIH record. 

The original plan also requires manual verification through Streamlit, including signal processing controls, classification and PDF generation. 

# **60. DOCUMENTATION** 

README.md must contain: 

1. Project overview 

2. Features 

3. Architecture 

4. Technologies 

5. Installation 

6. Dataset 

7. Classification scheme 

8. Running instructions 

9. Training instructions 

10. Model evaluation 

11. Screenshots 

12. Project limitations 

13. Medical disclaimer 

14. Future improvements 

# **61. IMPORTANT MEDICAL DISCLAIMER** 

The application must clearly display: 

**This system is developed for academic and educational purposes. ECG classifications are generated by machine-learning models and should not be interpreted as a medical diagnosis. The system is not a substitute for evaluation by a qualified healthcare professional.** 

# **62. FUTURE EXTENSIONS** 

These should **not be required for Version 2 completion** , but can be mentioned as future work: 

- Deep learning CNN 

- LSTM 

- Transformer-based ECG models 

- Real-time ECG streaming 

- Wearable device integration 

- Multi-lead ECG 

- Cloud deployment 

- Mobile application 

- Advanced SHAP explanations 

- Federated learning 

- Larger ECG datasets 

# **63. CLAUDE DEVELOPMENT RULES** 

This section is especially important if you're giving the plan to Claude. 

Claude must follow these rules: 

**Rule 1** 

# **Inspect the existing repository before changing anything.** 

**Rule 2** 

Treat: 

implementation_plan.md 

as the authoritative functional specification. 

# **Rule 3** 

Treat the existing repository as the current implementation. 

# **Rule 4** 

Do not unnecessarily rewrite working modules. 

# **Rule 5** 

Do not create duplicate files or duplicate functionality. 

# **Rule 6** 

Before modifying a module, understand how other modules depend on it. 

# **Rule 7** 

Maintain backward compatibility wherever practical. 

# **Rule 8** 

Do not silently change the classification scheme. 

# **Rule 9** 

Do not train models using test data. 

# **Rule 10** 

Do not perform preprocessing that learns parameters using the test dataset. 

# **Rule 11** 

Synthetic ECG must remain separate from final MIT-BIH evaluation. 

# **Rule 12** 

Do not claim clinical diagnostic accuracy. 

# **Rule 13** 

After every major implementation phase: 

Run tests 

Check imports 

Check application startup 

Verify affected functionality 

# **Rule 14** 

If an implementation decision conflicts with this plan, explain the conflict before changing the architecture. 

# **Rule 15** 

Prefer a **working simple implementation** over unnecessarily complicated architecture. 

# **64. FINAL PROJECT ARCHITECTURE** 

The final system should conceptually operate as: 

┌──────────────────┐ 

│   ECG INPUT      │ 

│ CSV / WFDB / Demo│ └──────── `┬` ─────────┘ │ 

▼ ┌──────────────────┐ 

│ DATA VALIDATION   │ └──────── `┬` ─────────┘ │ ▼ ┌──────────────────┐ 

│ SIGNAL QUALITY   │ 

│      SQI         │ └──────── `┬` ─────────┘ │ ▼ ┌──────────────────┐ 

│  PREPROCESSING   │ │ HP / NOTCH / BP  │ └──────── `┬` ─────────┘ │ ▼ ┌──────────────────┐ │   DSP ANALYSIS   │ │ Peaks / RR / FFT │ └──────── `┬` ─────────┘ │ 

▼ 



<!-- Start of picture text -->
                    ┌──────────────────┐<br>                    │ FEATURE ENGINE   │<br>                    │ Time/Morph/Stat  │<br>                    │ /Spectral        │<br>                    └──────── ┬ ─────────┘<br>                             │<br>▼<br>               ┌─────────────────────────────┐<br>               │       ML CLASSIFIERS        │<br>               │ LR | SVM | RF | XGBoost     │<br>               └────────────── ┬ ──────────────┘<br>                              │<br>▼<br>                    ┌──────────────────┐<br>                    │   PREDICTION     │<br>                    │ Class + Confidence│<br>                    └──────── ┬ ─────────┘<br>                             │<br>┴<br>                    ┌──────── ────────┐<br>▼ ▼<br>          ┌────────────────┐  ┌────────────────┐<br>          │ EXPLAINABLE AI │  │ VISUALIZATION  │<br>          └──────── ┬ ───────┘  └────────────────┘<br>                   │<br>┴<br>          ┌──────── ─────────┐<br>▼ ▼<br>   ┌──────────────┐   ┌──────────────┐<br>   │   SQLITE     │   │  PDF REPORT  │<br>   │   HISTORY    │   │   GENERATOR  │<br>   └──────────────┘   └──────────────┘<br><!-- End of picture text -->

# **65. THE FINAL GOAL** 

At the end, a professor should be able to open the application and see: 

INTELLIGENT ECG 

SIGNAL ANALYSIS 

┌───────────────────────┐ 

│     Upload ECG        │ 

└─────────── `┬` ───────────┘ 

↓ 

Signal Quality 

↓ 

DSP Processing 

↓ 

R-Peak Detection 

↓ 

Feature Extraction 

↓ 

ML Prediction 

↓ 

┌────────────────────────────┐ 

│ Prediction: VENTRICULAR    │ 

│ Confidence: 93.4%          │ 

└────────────────────────────┘ 

↓ 

Explainable AI 

↓ 

Model Performance 

↓ 

Analysis History 

↓ 

PDF Report 

The result should demonstrate **DSP + signal processing + feature engineering + classical machine learning + explainability + database + interactive visualization + automated reporting** in one integrated system. 

