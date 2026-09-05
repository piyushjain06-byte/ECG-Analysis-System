"""
Central configuration — single source of truth for anything that must stay
consistent across training, inference, and explainability (Implementation
Plan V2 §16 "feature vector metadata must be centralized", and the same
principle already applied to the AAMI mapping in §3).

Do not redefine FEATURE_COLS, CLASSIFICATION_VERSION, or FEATURE_VERSION
anywhere else in the codebase — import them from here.
"""

# --- Feature vector (fixed order — changing this changes what a saved model expects) ---
# Each entry: (name, description) for reproducibility/explainability metadata (plan §16).
FEATURE_SPEC = [
    ("pre_rr", "Preceding RR interval (s)"),
    ("post_rr", "Succeeding RR interval (s)"),
    ("rr_ratio", "pre_rr / post_rr"),
    ("local_rr_mean", "Mean RR interval over the +/-5 surrounding beats (s)"),
    ("mean", "Mean amplitude of the beat segment"),
    ("variance", "Variance of the beat segment"),
    ("std", "Standard deviation of the beat segment"),
    ("skewness", "Skewness of the beat segment"),
    ("kurtosis", "Kurtosis of the beat segment"),
    ("energy", "Sum of squared amplitude of the beat segment"),
    ("r_height", "Amplitude at the R-peak"),
    ("q_depth", "Minimum amplitude in the pre-R-peak Q window"),
    ("s_depth", "Minimum amplitude in the post-R-peak S window"),
    ("qrs_amplitude", "R height minus min(Q depth, S depth)"),
    ("qrs_duration_est", "Estimated QRS duration from the Q/S search window (s)"),
    ("dominant_frequency", "Dominant frequency of the beat segment (Hz)"),
    ("spectral_entropy", "Spectral entropy of the beat segment"),
]
FEATURE_COLS = [name for name, _ in FEATURE_SPEC]

# --- Classification scheme (plan §3) ---
CLASS_LABELS = ["N", "S", "V", "F"]

# --- Official dataset policy (plan decisions, this phase) ---
# The MIT-BIH Arrhythmia Database is the ONLY dataset used for the official,
# reported model performance. Synthetic ECG is for demo / UI / offline use
# and for an explicitly separate, clearly-labeled experiment only — it must
# never be pooled with MIT-BIH beats in the metrics that get reported as
# "the model's performance."
OFFICIAL_TRAINING_SOURCE = "mitbih"
ALLOWED_TRAINING_SOURCES = ("mitbih", "synthetic_experiment")

# --- Versioning (plan §25) ---
MODEL_VERSION = "v1.0"
DATASET_VERSION = "MIT-BIH-1.0.0"
FEATURE_VERSION = "v1"
CLASSIFICATION_VERSION = "4-class-AAMI"
