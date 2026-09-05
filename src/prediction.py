import os
import pickle
import numpy as np
import pandas as pd

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

from src.feature_extraction import extract_beat_features
from src.config import FEATURE_COLS


def _cache_resource(func):
    """Use st.cache_resource when running inside Streamlit; a no-op
    passthrough otherwise (e.g. under pytest) so this module has no hard
    Streamlit dependency for testing."""
    if _HAS_STREAMLIT:
        return st.cache_resource(func)
    return func


@_cache_resource
def load_prediction_utilities():
    """Loads the Standard Scaler and Label Encoder needed for consistent scaling.

    Cached with st.cache_resource so the pickle files are read from disk once
    per process instead of on every prediction call (plan §52/§54)."""
    scaler_path = "models/trained/scaler.pkl"
    le_path = "models/trained/label_encoder.pkl"

    if not os.path.exists(scaler_path) or not os.path.exists(le_path):
         return None, None

    with open(scaler_path, "rb") as f:
         scaler = pickle.load(f)

    with open(le_path, "rb") as f:
         label_encoder = pickle.load(f)

    return scaler, label_encoder


@_cache_resource
def load_classifier(model_name: str):
    """Loads a pre-trained pickle model based on standard name mapping.

    Cached with st.cache_resource, keyed on model_name, so switching models
    in the UI a second time does not re-read the file from disk."""
    filename = model_name.replace(" ", "_").lower() + ".pkl"
    filepath = os.path.join("models/trained", filename)

    if not os.path.exists(filepath):
         return None

    with open(filepath, "rb") as f:
         clf = pickle.load(f)
    return clf


def predict_ecg_beats(signal: np.ndarray, peaks: np.ndarray, fs: float = 360.0, model_name: str = "Random Forest"):
    """
    Given a clean ECG signal and detected QRS peak indices, extracts beat-by-beat features,
    scales features, runs predictions and provides classification labels and confidence levels.
    """
    scaler, label_encoder = load_prediction_utilities()
    clf = load_classifier(model_name)

    if scaler is None or label_encoder is None or clf is None:
         # Model files don't exist yet, return empty
         return None

    # Extract features for all peaks
    feat_df = extract_beat_features(signal, peaks, fs=fs)
    if feat_df.empty:
         return None

    X = feat_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # Run predictions
    preds_int = clf.predict(X_scaled)
    probs = clf.predict_proba(X_scaled)

    preds_label = label_encoder.inverse_transform(preds_int)

    # Package results into a final DataFrame
    feat_df['predicted_label'] = preds_label
    feat_df['confidence'] = np.max(probs, axis=1)

    # Map raw probability arrays
    for idx, class_name in enumerate(label_encoder.classes_):
         feat_df[f'prob_{class_name}'] = probs[:, idx]

    return feat_df
