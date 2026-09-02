"""End-to-end analysis used by the UI, PDF path, and screenshot generator."""

from src.signal_quality import assess_signal_quality
from src.preprocessing import full_dsp_pipeline, validate_ecg_signal
from src.signal_analysis import (
    pan_tompkins_detector,
    analyze_rr_intervals,
    compute_fft_spectrum,
    compute_spectral_energy_bands,
)
from src.feature_extraction import extract_beat_features
from src.prediction import predict_ecg_beats


CLASS_NAMES = {
    "N": "Normal sinus beat (N)",
    "V": "Ventricular ectopic / PVC (V)",
    "S": "Supraventricular ectopic / APC (S)",
    "F": "Fusion beat (F)",
}


def run_full_analysis(
    sig_raw,
    fs,
    use_baseline=True,
    baseline_method="butterworth",
    baseline_cutoff=0.5,
    use_notch=True,
    notch_freq=50.0,
    use_bandpass=True,
    bp_low=0.5,
    bp_high=40.0,
    bp_order=4,
    model_name="Random Forest",
):
    sig_raw = validate_ecg_signal(sig_raw, fs)
    quality = assess_signal_quality(sig_raw, fs=fs)
    sig_clean, stages = full_dsp_pipeline(
        sig_raw,
        fs=fs,
        use_baseline=use_baseline,
        baseline_method=baseline_method,
        baseline_cutoff=baseline_cutoff,
        use_notch=use_notch,
        notch_freq=notch_freq,
        use_bandpass=use_bandpass,
        bp_low=bp_low,
        bp_high=bp_high,
        bp_order=bp_order,
        normalization=None,
        return_stages=True,
    )
    peaks, pt_stages = pan_tompkins_detector(sig_clean, fs=fs)
    rr = analyze_rr_intervals(peaks, fs=fs)
    freqs_raw, mag_raw, dom_raw = compute_fft_spectrum(sig_raw, fs=fs)
    freqs, mag_clean, dom_clean = compute_fft_spectrum(sig_clean, fs=fs)
    bands_raw = compute_spectral_energy_bands(freqs_raw, mag_raw)
    bands_clean = compute_spectral_energy_bands(freqs, mag_clean)
    feat_df = extract_beat_features(sig_clean, peaks, fs=fs)
    pred_df = predict_ecg_beats(sig_clean, peaks, fs=fs, model_name=model_name)

    dominant = None
    confidence = None
    if pred_df is not None and not pred_df.empty:
        dominant = pred_df["predicted_label"].mode()[0]
        confidence = float(pred_df.loc[pred_df["predicted_label"] == dominant, "confidence"].mean())

    return {
        "quality": quality,
        "sig_raw": sig_raw,
        "stages": stages,
        "sig_clean": sig_clean,
        "peaks": peaks,
        "pt_stages": pt_stages,
        "rr": rr,
        "fft_freqs": freqs,
        "fft_mags": mag_clean,
        "fft_mags_raw": mag_raw,
        "fft_freqs_raw": freqs_raw,
        "dominant_freq": dom_clean,
        "dominant_freq_raw": dom_raw,
        "bands_raw": bands_raw,
        "bands_clean": bands_clean,
        "feat_df": feat_df,
        "pred_df": pred_df,
        "dominant_class": dominant,
        "confidence": confidence,
        "model_name": model_name,
    }