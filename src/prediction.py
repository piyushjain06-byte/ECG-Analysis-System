import os
import pickle
import numpy as np
import pandas as pd

from src.feature_extraction import extract_beat_features

def load_prediction_utilities():
    """Loads the Standard Scaler and Label Encoder needed for consistent scaling."""
    scaler_path = "models/trained/scaler.pkl"
    le_path = "models/trained/label_encoder.pkl"
    
    if not os.path.exists(scaler_path) or not os.path.exists(le_path):
         return None, None
         
    with open(scaler_path, "rb") as f:
         scaler = pickle.load(f)
         
    with open(le_path, "rb") as f:
         label_encoder = pickle.load(f)
         
    return scaler, label_encoder


def load_classifier(model_name: str):
    """Loads a pre-trained pickle model based on standard name mapping."""
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
         
    feature_cols = [
        'pre_rr', 'post_rr', 'rr_ratio', 'local_rr_mean',
        'mean', 'variance', 'std', 'skewness', 'kurtosis', 'energy',
        'r_height', 'q_depth', 's_depth', 'qrs_amplitude', 'qrs_duration_est',
        'dominant_frequency', 'spectral_entropy'
    ]
    
    X = feat_df[feature_cols].values
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
