"""
Model training.

IMPORTANT (decisions locked in for Implementation Plan V2):
  - The MIT-BIH Arrhythmia Database is the ONLY dataset used to produce the
    official, reported model performance (`source="mitbih"`, the default).
  - The synthetic generator may be used for a SEPARATE, clearly-labeled
    experiment (`source="synthetic_experiment"`) for demo/testing/offline
    use — it is never pooled with MIT-BIH beats in one train/test split.
    There is deliberately no "both" option any more: mixing the two for the
    numbers that get reported as "the model's performance" is exactly what
    the plan forbids.
"""

import os
import json
import pickle
import platform
import time
import warnings
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, f1_score, roc_auc_score, average_precision_score,
)
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

from src.data_loader import generate_synthetic_ecg
from src.preprocessing import full_dsp_pipeline
from src.feature_extraction import extract_beat_features
from src.mitbih_data import generate_ml_features_dataset_from_mitbih, available_mitbih_records
from src.config import (
    FEATURE_COLS, MODEL_VERSION, DATASET_VERSION, FEATURE_VERSION,
    CLASSIFICATION_VERSION, ALLOWED_TRAINING_SOURCES, OFFICIAL_TRAINING_SOURCE,
)

os.makedirs("models/trained", exist_ok=True)
os.makedirs("models/metadata", exist_ok=True)


def generate_ml_features_dataset(num_records: int = 24, fs: float = 360.0, seed_base: int = 0):
    """
    Beat-level features with expert-style labels from the synthetic generator.
    Uses annotated R locations (not detector output) so minority classes are not dropped.
    Record IDs stay attached for record-level train/test split (no patient leakage).

    This is ONLY used for the synthetic_experiment training source, or for
    demo/testing purposes elsewhere in the app - never pooled with MIT-BIH.

    seed_base fixes the global numpy RNG before the per-record heart-rate draw
    so two calls with identical arguments produce an identical dataset
    (reproducibility, plan §50; also required for save/load-round-trip tests
    that regenerate the same dataset independently).
    """
    np.random.seed(seed_base)
    all_features = []
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


def _record_level_split_3way(dataset: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """
    Patient/record-level 70/15/15 train/validation/test split (plan §17).
    Every beat from one record stays entirely in exactly one split. Records
    that contain any non-N beat are shuffled separately from pure-N records
    so arrhythmia classes are represented across all three splits where the
    record count allows it.
    """
    rng = np.random.RandomState(seed)
    recs = dataset.groupby("record_id")["label"].apply(lambda s: set(s.unique()))
    arr_recs = [r for r, labs in recs.items() if labs - {"N"}]
    nsr_recs = [r for r in recs.index if r not in arr_recs]
    rng.shuffle(arr_recs)
    rng.shuffle(nsr_recs)

    def _split_group(records):
        n = len(records)
        if n == 0:
            return [], [], []
        n_test = max(1, round(n * test_frac)) if n >= 3 else (1 if n > 1 else 0)
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        n_test = min(n_test, n - 1) if n > 1 else 0
        n_val = min(n_val, max(0, n - n_test - 1))
        test = records[:n_test]
        val = records[n_test:n_test + n_val]
        train = records[n_test + n_val:]
        return train, val, test

    arr_train, arr_val, arr_test = _split_group(arr_recs)
    nsr_train, nsr_val, nsr_test = _split_group(nsr_recs)

    train_records = np.array(arr_train + nsr_train)
    val_records = np.array(arr_val + nsr_val)
    test_records = np.array(arr_test + nsr_test)

    # Guarantee at least one record in val/test if there were enough records overall.
    if len(val_records) == 0 and len(train_records) > 1:
        val_records = train_records[-1:]
        train_records = train_records[:-1]
    if len(test_records) == 0 and len(train_records) > 1:
        test_records = train_records[-1:]
        train_records = train_records[:-1]

    return train_records, val_records, test_records


def _bounded_search_space():
    """Small, bounded hyperparameter grids (plan §21) — enough to beat naive
    defaults without turning training into a long-running job (Rule 15)."""
    return {
        "Logistic Regression": {"C": [0.1, 1.0, 3.0], "solver": ["lbfgs", "liblinear"]},
        "Support Vector Machine": {"C": [0.5, 1.0, 3.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        "Random Forest": {"n_estimators": [100, 180, 250], "max_depth": [8, 12, 16], "min_samples_split": [2, 5]},
        "XGBoost": {
            "n_estimators": [120, 180, 250],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.05, 0.08, 0.12],
            "subsample": [0.8, 0.9, 1.0],
        },
    }


def _tune_hyperparameters(model_name, base_estimator, X_train, y_train, groups_train, sample_weight=None):
    """
    RandomizedSearchCV with patient-aware CV (GroupKFold on record_id), using
    TRAINING DATA ONLY (plan §18/§21). Falls back to the untuned estimator if
    there are too few distinct records to form CV folds (small demo datasets)
    or if the search raises for any reason - tuning is a nice-to-have, it must
    never crash the training run.
    """
    n_groups = len(np.unique(groups_train))
    n_splits = min(3, n_groups)
    if n_splits < 2:
        print(f"  [{model_name}] Skipping hyperparameter search - only {n_groups} training record(s) available for patient-aware CV.")
        return base_estimator, {}

    param_dist = _bounded_search_space()[model_name]
    cv = GroupKFold(n_splits=n_splits)
    fit_params = {}
    if model_name == "XGBoost" and sample_weight is not None:
        fit_params["sample_weight"] = sample_weight

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            search = RandomizedSearchCV(
                base_estimator,
                param_distributions=param_dist,
                n_iter=min(6, int(np.prod([len(v) for v in param_dist.values()]))),
                cv=cv,
                scoring="f1_macro",
                random_state=42,
                n_jobs=1,
            )
            search.fit(X_train, y_train, groups=groups_train, **fit_params)
        return search.best_estimator_, search.best_params_
    except Exception as e:
        print(f"  [{model_name}] Hyperparameter search failed ({e}); using default parameters.")
        return base_estimator, {}


def _sensitivity_specificity(y_true, y_pred, labels_idx):
    """Per-class sensitivity (=recall) and specificity from the confusion
    matrix, then macro-averaged - reported as distinct metrics per plan §22,
    since 'recall' alone doesn't surface specificity."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    sens_list, spec_list = [], []
    total = cm.sum()
    for i in range(len(labels_idx)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(sens_list)), float(np.mean(spec_list))


def _roc_pr_auc(y_test, probs, labels_idx):
    """Macro ROC-AUC / PR-AUC (one-vs-rest). Returns (None, None) if a class
    is entirely absent from the test set (undefined, not a real 0)."""
    try:
        y_bin = label_binarize(y_test, classes=labels_idx)
        if y_bin.shape[1] == 1:  # binary-degenerate edge case
            y_bin = np.hstack([1 - y_bin, y_bin])
        roc_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
        pr_auc = average_precision_score(y_bin, probs, average="macro")
        return float(roc_auc), float(pr_auc)
    except Exception:
        return None, None


def train_eval_save_models(source: str = OFFICIAL_TRAINING_SOURCE, mitbih_dir: str = "data/mitbih",
                            num_synthetic: int = 24, tune_hyperparameters: bool = True,
                            models_dir: str = "models"):
    """
    source: "mitbih" (official - the only source used for reported
    performance) or "synthetic_experiment" (separate, clearly-labeled,
    never combined with MIT-BIH beats).

    models_dir: where trained/*.pkl and metadata/ml_performance.json are
    written. Defaults to "models" (the real app location); tests pass a
    temp directory so a test run never overwrites the official artifacts.
    """
    if source not in ALLOWED_TRAINING_SOURCES:
        raise ValueError(f"source must be one of {ALLOWED_TRAINING_SOURCES}, got '{source}'.")

    trained_dir = os.path.join(models_dir, "trained")
    metadata_dir = os.path.join(models_dir, "metadata")
    os.makedirs(trained_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    t0 = time.time()
    sources_used = {}

    if source == "mitbih":
        mitbih_available = available_mitbih_records(mitbih_dir)
        if not mitbih_available:
            raise FileNotFoundError(
                f"No MIT-BIH records found in '{mitbih_dir}'. Run `python download_mitbih.py` "
                f"first - the official model is trained on real MIT-BIH data only (see decision "
                f"log); synthetic data is not used as a silent substitute."
            )
        dataset = generate_ml_features_dataset_from_mitbih(mitbih_available, dest_dir=mitbih_dir)
        sources_used["mitbih_records"] = int(dataset["record_id"].nunique()) if not dataset.empty else 0
        sources_used["mitbih_beats"] = int(len(dataset))
    else:  # synthetic_experiment
        dataset = generate_ml_features_dataset(num_records=num_synthetic)
        sources_used["synthetic_records"] = int(dataset["record_id"].nunique()) if not dataset.empty else 0
        sources_used["synthetic_beats"] = int(len(dataset))
        print(
            "NOTE: this is the synthetic_experiment training run. These results are NOT "
            "the official reported model performance (that is MIT-BIH only) - see "
            "models/metadata/ml_performance.json's 'trained_on' field before quoting any number."
        )

    if dataset.empty:
        raise ValueError(f"Feature dataset generation failed or is empty for source='{source}'.")

    print(f"\nDataset generated. Shape: {dataset.shape}")
    print(f"Class distribution:\n{dataset['label'].value_counts().to_string()}")

    X = dataset[FEATURE_COLS].values
    y_raw = dataset["label"].values
    records = dataset["record_id"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_mapping = {int(i): str(name) for i, name in enumerate(label_encoder.classes_)}
    labels_idx = list(range(len(label_encoder.classes_)))

    train_records, val_records, test_records = _record_level_split_3way(dataset)
    train_mask = np.isin(records, train_records)
    val_mask = np.isin(records, val_records)
    test_mask = np.isin(records, test_records)

    X_train, y_train, groups_train = X[train_mask], y[train_mask], records[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nTrain: {len(X_train)} beats from {len(train_records)} record(s): {sorted(train_records)}")
    print(f"Val:   {len(X_val)} beats from {len(val_records)} record(s): {sorted(val_records)}")
    print(f"Test:  {len(X_test)} beats from {len(test_records)} record(s): {sorted(test_records)}")

    train_label_counts = pd.Series(label_encoder.inverse_transform(y_train)).value_counts().to_dict()
    test_label_counts = pd.Series(label_encoder.inverse_transform(y_test)).value_counts().to_dict() if len(y_test) else {}
    missing_in_test = [c for c in label_encoder.classes_ if test_label_counts.get(c, 0) == 0]
    missing_in_train = [c for c in label_encoder.classes_ if train_label_counts.get(c, 0) == 0]
    if missing_in_test:
        print(f"Note: class(es) {missing_in_test} have 0 beats in the test set - their metrics for that class are undefined, not a real 0.")
    if missing_in_train:
        print(f"Warning: class(es) {missing_in_train} have 0 beats in training - the model cannot learn them at all.")

    # --- Scaling: fit on TRAIN ONLY, reuse for val/test/inference (plan §9/§18) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) else X_val
    X_test_scaled = scaler.transform(X_test) if len(X_test) else X_test

    with open(os.path.join(trained_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(trained_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    sw_train = compute_sample_weight("balanced", y_train)

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=12, class_weight="balanced_subsample", random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, eval_metric="mlogloss", random_state=42,
        ),
    }

    performance = {}
    best_params_used = {}

    print("\nTraining models (hyperparameter search on TRAIN records only, validated on the held-out val split)...")
    for model_name, base_clf in base_models.items():
        if tune_hyperparameters:
            tuned_clf, best_params = _tune_hyperparameters(
                model_name, base_clf, X_train_scaled, y_train, groups_train,
                sample_weight=sw_train if model_name == "XGBoost" else None,
            )
        else:
            tuned_clf, best_params = base_clf, {}
        best_params_used[model_name] = best_params

        m0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            if model_name == "XGBoost":
                tuned_clf.fit(X_train_scaled, y_train, sample_weight=sw_train)
            else:
                tuned_clf.fit(X_train_scaled, y_train)
        training_time_s = time.time() - m0

        # Validation-set metrics (used only to pick/verify hyperparameters - never test).
        val_f1_macro = None
        if len(X_val_scaled):
            val_preds = tuned_clf.predict(X_val_scaled)
            val_f1_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

        # Final, reported metrics: TEST split only, touched exactly once here.
        preds = tuned_clf.predict(X_test_scaled) if len(X_test_scaled) else np.array([])
        probs = tuned_clf.predict_proba(X_test_scaled) if len(X_test_scaled) else np.zeros((0, len(labels_idx)))

        acc = accuracy_score(y_test, preds) if len(preds) else 0.0
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0) if len(preds) else (0, 0, 0, None)
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0)) if len(preds) else 0.0
        sensitivity, specificity = _sensitivity_specificity(y_test, preds, labels_idx) if len(preds) else (0.0, 0.0)
        roc_auc, pr_auc = _roc_pr_auc(y_test, probs, labels_idx) if len(preds) else (None, None)

        clf_report = classification_report(
            y_test, preds, labels=labels_idx, target_names=list(label_encoder.classes_),
            output_dict=True, zero_division=0,
        ) if len(preds) else {}
        c_matrix = confusion_matrix(y_test, preds, labels=labels_idx).tolist() if len(preds) else []

        performance[model_name] = {
            "accuracy": float(acc),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "macro_f1": macro_f1,
            "roc_auc_macro_ovr": roc_auc,
            "pr_auc_macro": pr_auc,
            "sensitivity_macro": sensitivity,
            "specificity_macro": specificity,
            "training_time_seconds": round(training_time_s, 3),
            "val_f1_macro": val_f1_macro,
            "best_params": best_params,
            "confusion_matrix": c_matrix,
            "report": clf_report,
        }
        model_filename = model_name.replace(" ", "_").lower() + ".pkl"
        with open(os.path.join(trained_dir, model_filename), "wb") as f:
            pickle.dump(tuned_clf, f)
        print(
            f"  {model_name:<24} acc={acc:.3f} macroF1={macro_f1:.3f} weightedF1={f1_w:.3f} "
            f"rocauc={roc_auc if roc_auc is not None else float('nan'):.3f} "
            f"train_time={training_time_s:.1f}s  params={best_params}"
        )

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION if source == "mitbih" else "synthetic-generator",
        "feature_version": FEATURE_VERSION,
        "classification_version": CLASSIFICATION_VERSION,
        "training_date": pd.Timestamp.now().isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "class_mapping": class_mapping,
        "feature_cols": FEATURE_COLS,
        "performance": performance,
        "class_distribution": dataset["label"].value_counts().to_dict(),
        "train_records": sorted(map(str, train_records)),
        "val_records": sorted(map(str, val_records)),
        "test_records": sorted(map(str, test_records)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "split_note": (
            "Record-level 70/15/15 train/validation/test split: all beats from one "
            "record/patient stay in exactly one split. Hyperparameter search used "
            "GroupKFold on training records only; test split is touched exactly once "
            "for the reported metrics above."
        ),
        "data_sources": sources_used,
        "trained_on": source,
        "is_official_result": source == OFFICIAL_TRAINING_SOURCE,
        "missing_classes_in_test": missing_in_test,
    }
    with open(os.path.join(metadata_dir, "ml_performance.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    if performance:
        best = max(performance.items(), key=lambda kv: kv[1]["macro_f1"])
        print(f"\nBest model on this split (by Macro F1, not raw accuracy): {best[0]} (Macro F1={best[1]['macro_f1']:.3f})")
        metadata["best_model"] = best[0]
        with open(os.path.join(metadata_dir, "ml_performance.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    print(f"\nModel training complete in {time.time()-t0:.1f}s. source='{source}' "
          f"({'OFFICIAL' if source == OFFICIAL_TRAINING_SOURCE else 'experimental, not reported as official'}). "
          f"Saved to models/trained/ and models/metadata/ml_performance.json.")
    return performance


if __name__ == "__main__":
    import sys
    import argparse

    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser(description="Train the ECG beat classifiers.")
    parser.add_argument(
        "--source", choices=list(ALLOWED_TRAINING_SOURCES), default=OFFICIAL_TRAINING_SOURCE,
        help="'mitbih' (default, official) = real MIT-BIH recordings only (needs data/mitbih/, "
             "run download_mitbih.py first). 'synthetic_experiment' = synthetic-only, for a "
             "separate experiment; never combined with MIT-BIH and never the official reported number.",
    )
    parser.add_argument("--mitbih-dir", default="data/mitbih")
    parser.add_argument("--num-synthetic", type=int, default=24,
                         help="Synthetic records to generate when --source synthetic_experiment.")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search (faster, uses fixed defaults).")
    args = parser.parse_args()

    train_eval_save_models(
        source=args.source, mitbih_dir=args.mitbih_dir,
        num_synthetic=args.num_synthetic, tune_hyperparameters=not args.no_tune,
    )
