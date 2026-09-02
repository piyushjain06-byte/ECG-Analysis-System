import os
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

from src.data_loader import generate_synthetic_ecg
from src.preprocessing import full_dsp_pipeline
from src.feature_extraction import extract_beat_features
from src.mitbih_data import generate_ml_features_dataset_from_mitbih, available_mitbih_records

os.makedirs("models/trained", exist_ok=True)
os.makedirs("models/metadata", exist_ok=True)

FEATURE_COLS = [
    "pre_rr", "post_rr", "rr_ratio", "local_rr_mean",
    "mean", "variance", "std", "skewness", "kurtosis", "energy",
    "r_height", "q_depth", "s_depth", "qrs_amplitude", "qrs_duration_est",
    "dominant_frequency", "spectral_entropy",
]


def generate_ml_features_dataset(num_records: int = 24, fs: float = 360.0):
    """
    Beat-level features with expert-style labels from the synthetic generator.
    Uses annotated R locations (not detector output) so minority classes are not dropped.
    Record IDs stay attached for record-level train/test split (no patient leakage).
    """
    all_features = []
    # More arrhythmia records so V/S/F appear in both train and test.
    rhythms = (
        ["arrhythmia"] * 12
        + ["normal"] * 4
        + ["bradycardia"] * 4
        + ["tachycardia"] * 4
    )
    rhythms = rhythms[:num_records]

    print(f"Generating features for {num_records} simulated records...")

    for r_idx in range(num_records):
        record_id = f"sim_{100 + r_idx}"
        rhythm_type = rhythms[r_idx]
        hr = float(np.random.randint(55, 105))
        if rhythm_type == "bradycardia":
            hr = 45.0
        elif rhythm_type == "tachycardia":
            hr = 118.0

        sig_raw, fs, ann_sample, ann_symbol = generate_synthetic_ecg(
            duration=50.0,
            fs=fs,
            heart_rate=hr,
            rhythm_type=rhythm_type,
            noise_level=np.random.uniform(0.02, 0.07),
            baseline_noise_level=np.random.uniform(0.08, 0.18),
            seed=42 + r_idx,
        )
        sig_clean = full_dsp_pipeline(sig_raw, fs=fs, normalization=None)
        if ann_sample is None or len(ann_sample) < 4:
            continue
        record_df = extract_beat_features(sig_clean, ann_sample, fs=fs, labels=list(ann_symbol))
        if not record_df.empty:
            record_df["record_id"] = record_id
            all_features.append(record_df)

    if not all_features:
        return pd.DataFrame()
    return pd.concat(all_features, ignore_index=True)


def _record_level_split(dataset: pd.DataFrame, test_frac: float = 0.25, seed: int = 42):
    """Keep all beats of a record on one side. Prefer mixing arrhythmia records into both sets."""
    rng = np.random.RandomState(seed)
    recs = dataset.groupby("record_id")["label"].apply(lambda s: set(s.unique()))
    arr_recs = [r for r, labs in recs.items() if labs - {"N"}]
    nsr_recs = [r for r, labs in recs.items() if r not in arr_recs]
    rng.shuffle(arr_recs)
    rng.shuffle(nsr_recs)

    n_test_arr = max(2, int(round(len(arr_recs) * test_frac)))
    n_test_nsr = max(1, int(round(len(nsr_recs) * test_frac)))
    test_records = arr_recs[:n_test_arr] + nsr_recs[:n_test_nsr]
    train_records = [r for r in recs.index if r not in test_records]
    return np.array(train_records), np.array(test_records)


def train_eval_save_models(source: str = "both", mitbih_dir: str = "data/mitbih", num_synthetic: int = None):
    """
    source: "synthetic" (offline, no real data needed), "mitbih" (real
    patient recordings only, requires data/mitbih/ populated via
    download_mitbih.py), or "both" (recommended - real data teaches true
    morphology/noise, synthetic tops up rare classes and guarantees the
    app always has something to train on).

    num_synthetic: how many synthetic records to add. If left as None,
    it's chosen automatically: 24 when there's no real data (synthetic
    carries the whole dataset), or 8 when combined with real MIT-BIH data
    (enough to top up rare classes without drowning out the real signal
    the whole point of "both" is to add).
    """
    t0 = time.time()
    datasets = []
    sources_used = {}

    mitbih_available = available_mitbih_records(mitbih_dir) if source in ("mitbih", "both") else []

    if num_synthetic is None:
        num_synthetic = 8 if (source == "both" and mitbih_available) else 24

    if source in ("synthetic", "both") and num_synthetic > 0:
        synth = generate_ml_features_dataset(num_records=num_synthetic)
        if not synth.empty:
            datasets.append(synth)
            sources_used["synthetic_records"] = int(synth["record_id"].nunique())
            sources_used["synthetic_beats"] = int(len(synth))

    if source in ("mitbih", "both"):
        if not mitbih_available:
            msg = (
                f"No MIT-BIH records found in '{mitbih_dir}'. Run "
                f"`python download_mitbih.py` first."
            )
            if source == "mitbih":
                raise FileNotFoundError(msg)
            print(f"Warning: {msg} Continuing with synthetic data only.")
        else:
            if len(mitbih_available) <= 8:
                print(
                    f"Note: only {len(mitbih_available)} MIT-BIH record(s) available "
                    f"(the quick set). Run `python download_mitbih.py --full` for the "
                    f"44-record AAMI benchmark set before reporting these numbers anywhere formal - "
                    f"a handful of records gives a noisy, small test set."
                )
            real = generate_ml_features_dataset_from_mitbih(mitbih_available, dest_dir=mitbih_dir)
            if not real.empty:
                datasets.append(real)
                sources_used["mitbih_records"] = int(real["record_id"].nunique())
                sources_used["mitbih_beats"] = int(len(real))

    if not datasets:
        raise ValueError("Feature dataset generation failed or is empty.")

    dataset = pd.concat(datasets, ignore_index=True)

    print(f"\nDataset generated. Shape: {dataset.shape}")
    print(f"Class distribution:\n{dataset['label'].value_counts().to_string()}")

    X = dataset[FEATURE_COLS].values
    y_raw = dataset["label"].values
    records = dataset["record_id"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_mapping = {int(i): str(name) for i, name in enumerate(label_encoder.classes_)}

    train_records, test_records = _record_level_split(dataset)
    train_mask = np.isin(records, train_records)
    test_mask = np.isin(records, test_records)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nTrain: {len(X_train)} beats from {len(train_records)} record(s): {sorted(train_records)}")
    print(f"Test:  {len(X_test)} beats from {len(test_records)} record(s): {sorted(test_records)}")
    train_label_counts = pd.Series(label_encoder.inverse_transform(y_train)).value_counts().to_dict()
    test_label_counts = pd.Series(label_encoder.inverse_transform(y_test)).value_counts().to_dict()
    print(f"Train labels: {train_label_counts}")
    print(f"Test labels:  {test_label_counts}")

    missing_in_test = [c for c in label_encoder.classes_ if test_label_counts.get(c, 0) == 0]
    missing_in_train = [c for c in label_encoder.classes_ if train_label_counts.get(c, 0) == 0]
    if missing_in_test:
        print(
            f"Note: class(es) {missing_in_test} have 0 beats in the test set - their "
            f"precision/recall will read as 0.00 (undefined, not a real failure). This "
            f"means the dataset doesn't have enough of that class spread across records yet."
        )
    if missing_in_train:
        print(f"Warning: class(es) {missing_in_train} have 0 beats in training - the model cannot learn them at all.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open("models/trained/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/trained/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    sw = compute_sample_weight("balanced", y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=12, class_weight="balanced_subsample", random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=42,
        ),
    }

    performance = {}
    labels_idx = list(range(len(label_encoder.classes_)))

    print("\nTraining models...")
    for model_name, clf in models.items():
        m0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            if model_name == "XGBoost":
                clf.fit(X_train_scaled, y_train, sample_weight=sw)
            else:
                clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        clf_report = classification_report(
            y_test,
            preds,
            labels=labels_idx,
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        )
        c_matrix = confusion_matrix(y_test, preds, labels=labels_idx).tolist()
        performance[model_name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": c_matrix,
            "report": clf_report,
        }
        model_filename = model_name.replace(" ", "_").lower() + ".pkl"
        with open(f"models/trained/{model_filename}", "wb") as f:
            pickle.dump(clf, f)
        print(f"  {model_name:<24} acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  ({time.time()-m0:.1f}s)")

    metadata = {
        "class_mapping": class_mapping,
        "feature_cols": FEATURE_COLS,
        "performance": performance,
        "class_distribution": dataset["label"].value_counts().to_dict(),
        "train_records": sorted(map(str, train_records)),
        "test_records": sorted(map(str, test_records)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_note": "Record-level split: all beats from one record/patient stay in train or test only.",
        "data_sources": sources_used,
        "trained_on": source,
        "missing_classes_in_test": missing_in_test,
    }
    with open("models/metadata/ml_performance.json", "w") as f:
        json.dump(metadata, f, indent=4)

    best = max(performance.items(), key=lambda kv: kv[1]["f1_score"])
    print(f"\nBest model on this split: {best[0]} (F1={best[1]['f1_score']:.3f})")

    best_report = best[1]["report"]
    class_names = list(label_encoder.classes_)
    print(f"\nPer-class breakdown for {best[0]} (the number that actually matters, not the weighted average):")
    print(f"  {'Class':<8}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for c in class_names:
        row = best_report.get(c, {})
        print(f"  {c:<8}{row.get('precision', 0):>10.3f}{row.get('recall', 0):>10.3f}{row.get('f1-score', 0):>10.3f}{int(row.get('support', 0)):>10d}")
    print(f"\nConfusion matrix for {best[0]} (rows = actual, columns = predicted, order = {class_names}):")
    for row_label, row in zip(class_names, best[1]["confusion_matrix"]):
        print(f"  {row_label:<4} {row}")

    print(f"\nModel training complete in {time.time()-t0:.1f}s. Saved to models/trained/ and models/metadata/ml_performance.json.")
    return performance


if __name__ == "__main__":
    import sys
    import argparse

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser(description="Train the ECG beat classifiers.")
    parser.add_argument(
        "--source", choices=["synthetic", "mitbih", "both"], default="both",
        help="synthetic = offline only, mitbih = real MIT-BIH recordings only "
             "(needs data/mitbih/, run download_mitbih.py first), both = recommended.",
    )
    parser.add_argument("--mitbih-dir", default="data/mitbih")
    parser.add_argument(
        "--num-synthetic", type=int, default=None,
        help="Number of synthetic records to add. Default: auto (8 when combined with real "
             "data so it doesn't drown out the real signal, 24 when synthetic-only). Pass 0 "
             "to disable synthetic data entirely.",
    )
    args = parser.parse_args()

    train_eval_save_models(source=args.source, mitbih_dir=args.mitbih_dir, num_synthetic=args.num_synthetic)