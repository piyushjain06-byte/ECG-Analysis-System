# CardioDSP Lab — Setup & Run

## 1. Create a virtual environment
```
python -m venv venv
```
Windows:
```
venv\Scripts\activate
```
macOS / Linux:
```
source venv/bin/activate
```

## 2. Install dependencies
```
pip install -r requirements.txt
```

## 3. Run the app
```
streamlit run app.py
```
This opens the app at http://localhost:8501

The app ships with models already trained on the synthetic generator, so it
works fully offline out of the box.

## 4. Train on real ECG recordings (MIT-BIH Arrhythmia Database)

The models can also be trained on real, cardiologist-annotated patient
recordings from PhysioNet's MIT-BIH Arrhythmia Database, instead of just
the synthetic generator.

**Step 1 — download real recordings (needs internet):**
```
python download_mitbih.py            # quick set: 8 records, ~15 MB
python download_mitbih.py --full      # full AAMI benchmark set: 44 records, ~100 MB
```
Files are saved to `data/mitbih/`. Safe to re-run — already-downloaded
records are skipped.

**Step 2 — retrain the models:**
```
python -m src.ml_training --source both       # real + synthetic (recommended)
python -m src.ml_training --source mitbih      # real recordings only
python -m src.ml_training --source synthetic   # synthetic only (original behavior)
```
This overwrites `models/trained/*.pkl` and `models/metadata/ml_performance.json`.
Restart the Streamlit app afterward to pick up the new models.

Notes:
- Beats are labeled from the actual cardiologist annotations shipped with
  each record (not re-detected), mapped to the AAMI EC57 classes N / S / V / F.
- The train/test split is done **per patient record**, so no patient's
  beats appear in both sets (avoids inflated accuracy from leakage).
- `models/metadata/ml_performance.json` records which data sources and
  record IDs were actually used for the run that produced the current models.

## Project structure
- `app.py` — Streamlit entry point, page routing, theme
- `src/` — DSP pipeline, feature extraction, ML training/prediction, PDF report, UI pages
- `src/mitbih_data.py` — MIT-BIH download/label-mapping/feature-extraction helpers
- `download_mitbih.py` — one-command script to fetch real recordings
- `data/sample/` — demo ECG CSV
- `data/mitbih/` — downloaded real recordings land here (not committed to git)
- `models/` — trained classifiers + metadata (ships pre-trained on synthetic data)
- `reports/` — generated PDF reports land here
- `screenshots/` — reference screenshots of each page
- `tests/` — pytest suite (`pytest tests/`)

## Other docs
- [README.md](README.md) — architecture, project overview, viva Q&A
- [DEPLOYMENT.md](DEPLOYMENT.md) — get a public Streamlit Cloud link
