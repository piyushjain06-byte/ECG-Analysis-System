import numpy as np
from scipy import signal as scipy_signal

def assess_signal_quality(signal: np.ndarray, fs: float = 360.0):
    """
    Assesses the quality of an ECG signal based on physiological and mathematical metrics:
    1. Fraction of missing or NaN values.
    2. Saturation / Flatlines (constant values in succession).
    3. Extreme amplitude violations (out of standard physical range of -5 mV to +5 mV).
    4. High-frequency noise level.
    5. Baseline Wander severity.

    Returns a dictionary of metrics and a final quality string: 'GOOD', 'ACCEPTABLE', or 'POOR'.
    Also returns a numeric 'sqi_score' (0-100) summarizing the same signals as a single number.
    """
    metrics = {}

    # Handle empty or invalid shape
    if signal is None or len(signal) == 0:
        return {
            'status': 'POOR',
            'sqi_score': 0,
            'missing_ratio': 1.0,
            'flatline_ratio': 1.0,
            'extreme_ratio': 0.0,
            'hf_noise_ratio': 1.0,
            'details': "Empty signal received."
        }
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be a finite positive number.")

    signal = np.asarray(signal)
    total_len = len(signal)

    # 1. Missing Data
    nan_count = np.sum(np.isnan(signal)) + np.sum(np.isinf(signal))
    missing_ratio = float(nan_count / total_len)
    metrics['missing_ratio'] = missing_ratio

    # Clean up NaNs internally for subsequent computations
    clean_sig = np.copy(signal)
    if nan_count > 0:
        if nan_count == total_len:
            return {
                'status': 'POOR',
                'sqi_score': 0,
                'missing_ratio': 1.0,
                'flatline_ratio': 0.0,
                'extreme_ratio': 0.0,
                'hf_noise_ratio': 0.0,
                'details': "Signal is completely composed of NaN/Inf values."
            }
        mean_val = np.nanmean(signal) if not np.isnan(np.nanmean(signal)) else 0.0
        clean_sig[np.isnan(clean_sig) | np.isinf(clean_sig)] = mean_val

    # 2. Flatline / Saturation: only count repeated runs of at least 10 samples.
    flat_diff = np.diff(clean_sig) == 0.0
    flatline_samples = np.zeros(total_len, dtype=bool)
    if np.any(flat_diff):
        starts = np.flatnonzero(flat_diff & np.r_[True, ~flat_diff[:-1]])
        ends = np.flatnonzero(flat_diff & np.r_[~flat_diff[1:], True])
        for start, end in zip(starts, ends):
            if end - start + 2 >= 10:
                flatline_samples[start:end + 2] = True
    flatline_ratio = float(np.mean(flatline_samples))
    metrics['flatline_ratio'] = flatline_ratio

    # 3. Extreme Amplitudes (physiologically, ECGs rarely exceed 4.0 mV or are less than -4.0 mV)
    # Standard MIT-BIH recordings are calibrated, usually within -3 to 3 mV.
    extreme_count = np.sum((clean_sig > 4.5) | (clean_sig < -4.5))
    extreme_ratio = float(extreme_count / total_len)
    metrics['extreme_ratio'] = extreme_ratio

    # 4. High-frequency power fraction, estimated from a Welch PSD so the
    # result is interpretable and does not scale with the sampling rate.
    nperseg = min(total_len, max(8, int(fs * 4)))
    freqs, psd = scipy_signal.welch(clean_sig, fs=fs, nperseg=nperseg)
    total_power = float(np.sum(psd))
    hf_cutoff = min(40.0, 0.8 * (fs / 2.0))
    hf_power = float(np.sum(psd[freqs >= hf_cutoff]))
    hf_noise_ratio = hf_power / total_power if total_power > 1e-12 else 0.0
    metrics['hf_noise_ratio'] = hf_noise_ratio

    # 5. Baseline Drift Severity
    # Drift can be approximated by comparing a moving average with standard variance
    if len(clean_sig) > int(fs * 2):
        window_len = int(fs * 2)
        # Compute dynamic moving average using uniform filter or simply sum
        cumsum = np.cumsum(np.insert(clean_sig, 0, 0))
        moving_avg = (cumsum[window_len:] - cumsum[:-window_len]) / window_len
        drift_var = np.var(moving_avg)
        sig_var = np.var(clean_sig)
        if sig_var > 1e-6:
             baseline_drift_ratio = float(drift_var / sig_var)
        else:
             baseline_drift_ratio = 0.0
    else:
        baseline_drift_ratio = 0.0
    metrics['baseline_drift_ratio'] = baseline_drift_ratio

    # Final Decision logic
    status = 'GOOD'
    details = []

    if missing_ratio > 0.05:
        status = 'POOR'
        details.append(f"High ratio of missing data: {missing_ratio:.1%}")
    elif missing_ratio > 0.01:
        status = 'ACCEPTABLE'
        details.append(f"Minor missing data: {missing_ratio:.1%}")

    if flatline_ratio > 0.15:
        status = 'POOR'
        details.append(f"Significant flatlining/saturation: {flatline_ratio:.1%}")
    elif flatline_ratio > 0.03:
        if status != 'POOR':
            status = 'ACCEPTABLE'
        details.append(f"Minor flatlining: {flatline_ratio:.1%}")

    if extreme_ratio > 0.05:
        status = 'POOR'
        details.append(f"Extreme voltage spikes/artifacts: {extreme_ratio:.1%}")
    elif extreme_ratio > 0.005:
        if status != 'POOR':
            status = 'ACCEPTABLE'
        details.append(f"Minor voltage spikes: {extreme_ratio:.1%}")

    if hf_noise_ratio > 0.30:
        status = 'POOR'
        details.append("Excessive high-frequency power (muscle tremors / interference)")
    elif hf_noise_ratio > 0.10:
        if status != 'POOR':
            status = 'ACCEPTABLE'
        details.append("Significant high-frequency noise")

    if baseline_drift_ratio > 0.6:
        status = 'POOR'
        details.append("Severe baseline wander/drift")
    elif baseline_drift_ratio > 0.25:
        if status != 'POOR':
            status = 'ACCEPTABLE'
        details.append("Moderate baseline wander")

    # --- Numeric SQI score (0-100), derived from the same five signals above ---
    # Each component contributes a penalty capped so no single metric alone can
    # swing the score to zero; weights roughly mirror the severity thresholds above.
    penalty = 0.0
    penalty += min(35.0, missing_ratio * 700.0)          # 5% missing -> 35 pts
    penalty += min(50.0, flatline_ratio * (50.0 / 0.15))  # 15% flatline -> 50 pts
    penalty += min(20.0, extreme_ratio * 400.0)           # 5% extreme -> 20 pts
    penalty += min(20.0, max(0.0, hf_noise_ratio - 0.05) * 50.0)   # scales past a small baseline
    penalty += min(15.0, max(0.0, baseline_drift_ratio - 0.1) * 25.0)
    sqi_score = int(round(max(0.0, 100.0 - penalty)))
    metrics['sqi_score'] = sqi_score

    metrics['status'] = status
    metrics['details'] = "; ".join(details) if details else "Signal shows low noise and high consistency."

    return metrics
