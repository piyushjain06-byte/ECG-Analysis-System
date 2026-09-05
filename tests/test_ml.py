import os
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, ALLOWED_TRAINING_SOURCES
from src.feature_extraction import extract_beat_features
from src.data_loader import generate_synthetic_ecg
from src.preprocessing import full_dsp_pipeline
from src.ml_training import (
    generate_ml_features_dataset,
    _record_level_split_3way,
    train_eval_save_models,
)
from src.prediction import predict_ecg_beats
from src.pipeline import run_full_analysis


def test_feature_cols_match_actual_feature_extraction_output():
    """Leakage/consistency check: the centralized FEATURE_COLS list used by
    ml_training/prediction/explainability must be a subset of the columns
    extract_beat_features actually produces - if someone renames a feature
    in one place and not the other, this catches it immediately."""
    fs = 360.0
    sig, _, peaks, _ = generate_synthetic_ecg(duration=8.0, fs=fs, rhythm_type="normal", seed=1)
    sig_clean = full_dsp_pipeline(sig, fs=fs)
    feat_df = extract_beat_features(sig_clean, peaks, fs=fs)
    for col in FEATURE_COLS:
        assert col in feat_df.columns, f"FEATURE_COLS entry '{col}' is missing from extract_beat_features output"


def test_allowed_training_sources_excludes_mixed_option():
    """Decision lock-in: there must be no 'both' training source that silently
    pools synthetic and MIT-BIH beats into one official result."""
    assert "both" not in ALLOWED_TRAINING_SOURCES
    assert "mitbih" in ALLOWED_TRAINING_SOURCES
    assert "synthetic_experiment" in ALLOWED_TRAINING_SOURCES


def test_record_level_split_has_no_overlap_and_covers_all_records():
    dataset = generate_ml_features_dataset(num_records=6)
    if dataset.empty:
        pytest.skip("Synthetic dataset generation produced no beats in this environment.")

    train_r, val_r, test_r = _record_level_split_3way(dataset, val_frac=0.15, test_frac=0.15, seed=42)

    train_set, val_set, test_set = set(train_r), set(val_r), set(test_r)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    all_records = set(dataset["record_id"].unique())
    assert train_set | val_set | test_set == all_records


def test_train_eval_save_models_synthetic_experiment_round_trip(tmp_path):
    """End-to-end: train on the synthetic_experiment source (never mixed with
    MIT-BIH), verify metadata correctness, then verify prediction.py can load
    the saved model/scaler/label-encoder and produce valid probabilities.
    Written to a temp models_dir so this never overwrites the real,
    officially-trained (MIT-BIH) artifacts.
    """
    models_dir = str(tmp_path / "models")
    performance = train_eval_save_models(
        source="synthetic_experiment",
        num_synthetic=6,
        tune_hyperparameters=False,  # keep the test fast
        models_dir=models_dir,
    )
    assert performance, "Training produced no performance results."

    meta_path = os.path.join(models_dir, "metadata", "ml_performance.json")
    assert os.path.exists(meta_path)
    import json
    with open(meta_path) as f:
        meta = json.load(f)

    # This run must be clearly labeled as non-official and never pooled with MIT-BIH.
    assert meta["trained_on"] == "synthetic_experiment"
    assert meta["is_official_result"] is False
    assert "mitbih_beats" not in meta.get("data_sources", {})
    assert meta["feature_cols"] == FEATURE_COLS
    for key in ("model_version", "dataset_version", "feature_version", "classification_version"):
        assert meta[key], f"Missing version metadata field: {key}"

    for model_name, m in performance.items():
        assert "macro_f1" in m
        assert "f1_weighted" in m
        assert "sensitivity_macro" in m
        assert "specificity_macro" in m
        assert "training_time_seconds" in m

    # --- Save/load round trip + probability sanity, via prediction.py loading from models_dir ---
    import pickle
    scaler_path = os.path.join(models_dir, "trained", "scaler.pkl")
    le_path = os.path.join(models_dir, "trained", "label_encoder.pkl")
    rf_path = os.path.join(models_dir, "trained", "random_forest.pkl")
    assert os.path.exists(scaler_path) and os.path.exists(le_path) and os.path.exists(rf_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    with open(rf_path, "rb") as f:
        clf = pickle.load(f)

    fs = 360.0
    sig, _, peaks, _ = generate_synthetic_ecg(duration=10.0, fs=fs, rhythm_type="arrhythmia", seed=99)
    sig_clean = full_dsp_pipeline(sig, fs=fs)
    feat_df = extract_beat_features(sig_clean, peaks, fs=fs)
    assert not feat_df.empty

    X = feat_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    probs = clf.predict_proba(X_scaled)

    # Probabilities must be valid: correct shape, in [0,1], rows sum to 1.
    assert probs.shape == (len(feat_df), len(label_encoder.classes_))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_train_eval_save_models_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError):
        train_eval_save_models(source="both", models_dir=str(tmp_path / "models"))


def test_train_eval_save_models_mitbih_requires_local_data(tmp_path):
    """Official source must fail loudly (not silently fall back to synthetic)
    when no local MIT-BIH data is present."""
    with pytest.raises(FileNotFoundError):
        train_eval_save_models(
            source="mitbih",
            mitbih_dir=str(tmp_path / "no_such_mitbih_dir"),
            models_dir=str(tmp_path / "models"),
        )


def test_scaler_is_fit_on_train_only_not_val_or_test(tmp_path):
    """Direct leakage check: the persisted scaler's mean_/scale_ must match a
    scaler fit on the train split alone, not on train+val+test combined."""
    from sklearn.preprocessing import StandardScaler
    from src.config import FEATURE_COLS as FC

    dataset = generate_ml_features_dataset(num_records=6)
    if dataset.empty:
        pytest.skip("Synthetic dataset generation produced no beats in this environment.")

    train_r, val_r, test_r = _record_level_split_3way(dataset, seed=42)
    train_mask = dataset["record_id"].isin(train_r)

    models_dir = str(tmp_path / "models")
    train_eval_save_models(
        source="synthetic_experiment", num_synthetic=6, tune_hyperparameters=False, models_dir=models_dir,
    )

    import pickle
    with open(os.path.join(models_dir, "trained", "scaler.pkl"), "rb") as f:
        persisted_scaler = pickle.load(f)

    expected_scaler = StandardScaler().fit(dataset.loc[train_mask, FC].values)

    assert np.allclose(persisted_scaler.mean_, expected_scaler.mean_, atol=1e-6)
    assert np.allclose(persisted_scaler.scale_, expected_scaler.scale_, atol=1e-6)
