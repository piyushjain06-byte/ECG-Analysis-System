import numpy as np

def assess_signal_quality(signal: np.ndarray, fs: float = 360.0):
    """
    Assesses the quality of an ECG signal based on physiological and mathematical metrics:
    1. Fraction of missing or NaN values.
    2. Saturation / Flatlines (constant values in succession).
    3. Extreme amplitude violations (out of standard physical range of -5 mV to +5 mV).
    4. High-frequency noise level.
    5. Baseline Wander severity.
    
    Returns a dictionary of metrics and a final quality string: 'GOOD', 'ACCEPTABLE', or 'POOR'.
    """
    metrics = {}
    
    # Handle empty or invalid shape
    if signal is None or len(signal) == 0:
        return {
            'status': 'POOR',
            'missing_ratio': 1.0,
            'flatline_ratio': 1.0,
            'extreme_ratio': 0.0,
            'hf_noise_ratio': 1.0,
            'details': "Empty signal received."
        }
        
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
                'missing_ratio': 1.0,
                'flatline_ratio': 0.0,
                'extreme_ratio': 0.0,
                'hf_noise_ratio': 0.0,
                'details': "Signal is completely composed of NaN/Inf values."
            }
        mean_val = np.nanmean(signal) if not np.isnan(np.nanmean(signal)) else 0.0
        clean_sig[np.isnan(clean_sig) | np.isinf(clean_sig)] = mean_val

    # 2. Flatline / Saturation (where consecutive values are exactly the same)
    # Detect sequences of length >= 10 with identical values
    diff = np.diff(clean_sig)
    flat_diff = (diff == 0.0).astype(int)
    # Count how many samples are in flat segments
    flatline_ratio = float(np.sum(flat_diff) / total_len)
    metrics['flatline_ratio'] = flatline_ratio
    
    # 3. Extreme Amplitudes (physiologically, ECGs rarely exceed 4.0 mV or are less than -4.0 mV)
    # Standard MIT-BIH recordings are calibrated, usually within -3 to 3 mV.
    extreme_count = np.sum((clean_sig > 4.5) | (clean_sig < -4.5))
    extreme_ratio = float(extreme_count / total_len)
    metrics['extreme_ratio'] = extreme_ratio
    
    # 4. High-Frequency Noise (muscle artifact estimation)
    # We can estimate this by looking at high frequency components. 
    # A simple way of doing it in the time domain is the ratio of high-order difference variance
    # to standard signal variance, or using spectral estimates.
    if len(clean_sig) > 10:
        # High pass approximation: difference of difference
        hf_estimate = np.diff(clean_sig, n=2)
        hf_var = np.var(hf_estimate)
        sig_var = np.var(clean_sig)
        if sig_var > 1e-6:
            hf_noise_ratio = float(hf_var / sig_var)
        else:
            hf_noise_ratio = 1.0  # Signal has no variance
    else:
        hf_noise_ratio = 0.0
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
    
    if hf_noise_ratio > 0.4:
        status = 'POOR'
        details.append("Excessive high-frequency power (muscle tremors / interference)")
    elif hf_noise_ratio > 0.15:
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
        
    metrics['status'] = status
    metrics['details'] = "; ".join(details) if details else "Signal shows low noise and high consistency."
    
    return metrics
