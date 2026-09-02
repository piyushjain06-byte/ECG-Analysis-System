import numpy as np
import pandas as pd
import pickle
import os

from src.prediction import load_classifier, load_prediction_utilities

def get_model_feature_importance(model_name: str):
    """
    For tree-based models (Random Forest, XGBoost), retrieves standard Gini importance / Gain.
    """
    clf = load_classifier(model_name)
    _, label_encoder = load_prediction_utilities()
    
    if clf is None:
         return None
         
    feature_cols = [
        'pre_rr', 'post_rr', 'rr_ratio', 'local_rr_mean',
        'mean', 'variance', 'std', 'skewness', 'kurtosis', 'energy',
        'r_height', 'q_depth', 's_depth', 'qrs_amplitude', 'qrs_duration_est',
        'dominant_frequency', 'spectral_entropy'
    ]
    
    # Check if classifier has feature importances
    if hasattr(clf, 'feature_importances_'):
         importances = clf.feature_importances_
         feat_imp = pd.DataFrame({
             'feature': feature_cols,
             'importance': importances
         }).sort_values(by='importance', ascending=False)
         return feat_imp
    elif hasattr(clf, 'coef_'):
         # In Logistic Regression, retrieve weights
         coefs = clf.coef_
         # For multi-class, take mean absolute weight across classes
         mean_weights = np.mean(np.abs(coefs), axis=0)
         feat_imp = pd.DataFrame({
             'feature': feature_cols,
             'importance': mean_weights
         }).sort_values(by='importance', ascending=False)
         return feat_imp
    else:
         # SVM doesn't expose clean importances readily
         return None


def generate_clinical_explanation(beat_row: pd.Series):
    """
    Generates an expert system explanation based on standard clinical ECG criteria:
    - Normal (N): Stable RR intervals, standard amplitude, narrow QRS width.
    - Premature Ventricular Contraction / Ventricular (V): Wide QRS width, premature timing, discordant T/amplitude.
    - Atrial Premature Beat / Supraventricular (S): Premature timing, normal/narrow QRS width.
    - Fusion Beat (F): Intermediate QRS amplitude and width, premature timing.
    """
    label = beat_row.get('predicted_label', 'N')
    confidence = beat_row.get('confidence', 1.0)
    
    pre_rr = beat_row['pre_rr']
    post_rr = beat_row['post_rr']
    qrs_width = beat_row['qrs_duration_est']
    qrs_amp = beat_row['qrs_amplitude']
    dominant_frequency = beat_row['dominant_frequency']
    
    explanation = f"**Classification Verdict:** Beat was predicted as **'{label}'** with a model confidence of **{confidence:.1%}**.\n\n"
    
    contributions = []
    
    # 1. Evaluate timing criteria (pre-RR and post-RR)
    # Average normal heartbeat interval at 72 BPM is ~0.83 seconds
    if pre_rr < beat_row['local_rr_mean'] * 0.82:
         contributions.append(f"- **Timing: Early/Premature Beat**. Preceding RR interval of {pre_rr:.2f}s is significantly shorter than the local average interval of {beat_row['local_rr_mean']:.2f}s (shortened by {1.0 - (pre_rr/beat_row['local_rr_mean']):.1%}).")
         # Check for compensatory pause (common in PVCs)
         if post_rr > beat_row['local_rr_mean'] * 1.15:
              contributions.append(f"- **Timing: Compensatory Pause**. Succeeding RR interval of {post_rr:.2f}s is prolonged, indicating a cardiac reset pause.")
    else:
         contributions.append(f"- **Timing: Regular rhythm**. Preceding RR interval of {pre_rr:.2f}s is normal relative to local average pacemaker rate ({beat_row['local_rr_mean']:.2f}s).")
         
    # 2. Evaluate morphology criteria (QRS Width & Amplitude)
    # Physiologically, standard QRS is < 0.10s. Wide complexes are > 0.12s.
    if qrs_width >= 0.13:
         contributions.append(f"- **Morphology: Ventricular Prolongation**. Estimated QRS complex duration is {qrs_width*1000:.0f} ms. In clinical terms, a QRS duration exceeding 120 ms suggests sluggish electrical conduction through ventricular muscle sheets instead of normal fast conduction pathways.")
    else:
         contributions.append(f"- **Morphology: Narrow QRS complex**. QRS duration is {qrs_width*1000:.0f} ms, indicating normal, rapid electrical propagation through the bundle branches and Purkinje network.")
         
    if qrs_amp > 1.2:
         contributions.append(f"- **Morphology: High QRS amplitude**. Signal amplitude peak-to-peak of {qrs_amp:.2f} mV suggests high localized ventricular activation voltage.")
         
    # 3. Match clinical patterns to labels
    clinical_patterns = ""
    if label == 'V':
        clinical_patterns = (
            "**Physiological Reasoning:** This beat fits the profile of a **Premature Ventricular Contraction (PVC)**. "
            "The model observed an early ventricular depolarization (short preceding RR interval) which is wide (long QRS width) "
            "and often followed by a full compensatory pause. Because the impulse originates in the ventricles directly (rather than the SA node), "
            "conduction is cell-to-cell, causing a wider waveform."
        )
    elif label == 'S':
        clinical_patterns = (
            "**Physiological Reasoning:** This beat matches a **Supraventricular Ectopic Beat (APC)**. "
            "The beat occurs prematurely (short preceding RR interval), but the QRS morphology remains narrow and similar to a normal beat. "
            "This suggests the ectopic trigger originated in the atria above the ventricles, passing through the normal His-Purkinje conduction pathway."
        )
    elif label == 'F':
        clinical_patterns = (
            "**Physiological Reasoning:** This beat is classified as a **Fusion Beat**. "
            "A fusion beat occurs when a normal sinus impulse and an ectopic ventricular focus trigger ventricles simultaneously. "
            "Morphologically, it displays intermediate characteristics (moderate width and altered amplitude) that sit between a normal beat and a PVC."
        )
    else: # Normal
        clinical_patterns = (
            "**Physiological Reasoning:** This beat represents a **Normal Sinus Beat (N)**. "
            "Timing intervals are regular and QRS duration is narrow (<100ms), showcasing standard, healthy pacemaker pacing."
        )
        
    full_explanation = explanation + "### Extracted Features Contribution:\n" + "\n".join(contributions) + "\n\n" + clinical_patterns
    return full_explanation
