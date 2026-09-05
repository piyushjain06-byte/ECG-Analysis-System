import numpy as np
from scipy import signal as sig

def remove_baseline_wander(data: np.ndarray, fs: float = 360.0, method: str = 'butterworth', cutoff: float = 0.5):
    if method == 'butterworth':
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = sig.butter(N=1, Wn=normal_cutoff, btype='highpass', analog=False)
        filtered_data = sig.filtfilt(b, a, data)
        return filtered_data
    elif method == 'median':
        window_1 = int(0.2 * fs)
        window_2 = int(0.6 * fs)
        if window_1 % 2 == 0: window_1 += 1
        if window_2 % 2 == 0: window_2 += 1
        base_1 = sig.medfilt(data, kernel_size=window_1)
        baseline = sig.medfilt(base_1, kernel_size=window_2)
        return data - baseline
    elif method == 'detrend':
        t = np.arange(len(data))
        p = np.polyfit(t, data, 1)
        baseline = np.polyval(p, t)
        return data - baseline
    else:
        return data


def apply_notch_filter(data: np.ndarray, fs: float = 360.0, notch_freq: float = 50.0, quality_factor: float = 30.0):
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = sig.iirnotch(w0, quality_factor)
    filtered_data = sig.filtfilt(b, a, data)
    return filtered_data


def apply_bandpass_filter(data: np.ndarray, fs: float = 360.0, low_cut: float = 0.5, high_cut: float = 40.0, order: int = 4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    low = max(0.001, min(low, 0.99))
    high = max(0.001, min(high, 0.99))
    b, a = sig.butter(order, [low, high], btype='band')
    filtered_data = sig.filtfilt(b, a, data)
    return filtered_data


def normalize_signal(data: np.ndarray, method: str = 'minmax'):
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 1e-8:
            return (data - min_val) / (max_val - min_val)
        return data
    elif method == 'minmax_signed':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 1e-8:
             return 2.0 * ((data - min_val) / (max_val - min_val)) - 1.0
        return data
    elif method in ['standard', 'z-score', 'zscore']:
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 1e-8:
            return (data - mean_val) / std_val
        return data
    else:
        return data


def validate_ecg_signal(sig, fs, min_seconds: float = 2.0) -> np.ndarray:
    if sig is None:
        raise ValueError("No signal data was provided.")

    arr = np.asarray(sig)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    arr = arr.astype(np.float64, copy=False)

    if arr.size == 0:
        raise ValueError("The recording is empty (0 samples).")

    if fs is None or fs <= 0:
        raise ValueError(f"Invalid sampling frequency ({fs} Hz) - it must be a positive number.")

    min_samples = int(min_seconds * fs)
    if arr.size < min_samples:
        raise ValueError(
            f"Recording is too short ({arr.size / fs:.2f}s, {arr.size} samples). "
            f"At least {min_seconds:.0f}s is needed for the filters and beat detector to work reliably."
        )

    n_bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
    if n_bad > 0:
        raise ValueError(
            f"The signal contains {n_bad} missing or invalid value(s) (NaN/Inf). "
            f"Check the uploaded file for blank cells or non-numeric rows."
        )

    if np.std(arr) < 1e-9:
        raise ValueError("The signal is flat (no variation) - this doesn't look like an ECG recording.")

    return arr


def full_dsp_pipeline(data: np.ndarray, fs: float = 360.0,
                      use_baseline: bool = True, baseline_method: str = 'butterworth', baseline_cutoff: float = 0.5,
                      use_notch: bool = True, notch_freq: float = 50.0,
                      use_bandpass: bool = True, bp_low: float = 0.5, bp_high: float = 40.0, bp_order: int = 4,
                      normalization: str = None, return_stages: bool = False):
    processed = np.asarray(data, dtype=float).copy()
    stages = {"raw": processed.copy()}

    if use_baseline:
        processed = remove_baseline_wander(processed, fs, method=baseline_method, cutoff=baseline_cutoff)
    stages["baseline"] = processed.copy()

    if use_notch:
        processed = apply_notch_filter(processed, fs, notch_freq=notch_freq)
    stages["notch"] = processed.copy()

    if use_bandpass:
        processed = apply_bandpass_filter(processed, fs, low_cut=bp_low, high_cut=bp_high, order=bp_order)
    stages["bandpass"] = processed.copy()

    if normalization is not None:
        processed = normalize_signal(processed, method=normalization)
    stages["final"] = processed.copy()

    if return_stages:
        return processed, stages
    return processed
