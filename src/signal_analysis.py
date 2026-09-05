import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq

def pan_tompkins_detector(data: np.ndarray, fs: float = 360.0):
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = sig.butter(3, [low, high], btype='band')
    filt_ecg = sig.filtfilt(b, a, data)

    der_ecg = np.zeros_like(filt_ecg)
    der_ecg[4:] = (
        2.0 * filt_ecg[4:]
        + filt_ecg[3:-1]
        - filt_ecg[1:-3]
        - 2.0 * filt_ecg[:-4]
    ) / 8.0

    sq_ecg = der_ecg ** 2

    win_samples = int(0.150 * fs)
    if win_samples % 2 == 0:
        win_samples += 1

    integration_kernel = np.ones(win_samples) / win_samples
    mwi_ecg = np.convolve(sq_ecg, integration_kernel, mode='same')

    peak_indices = []

    min_dist = int(0.250 * fs)
    height_threshold = np.percentile(mwi_ecg, 90) * 0.3

    candidates, _ = sig.find_peaks(mwi_ecg, distance=min_dist, height=height_threshold)

    search_window = int(0.100 * fs)
    for cand in candidates:
        start_idx = max(0, cand - search_window)
        end_idx = min(len(data), cand + search_window)
        if start_idx < end_idx:
             local_peak = start_idx + np.argmax(np.abs(data[start_idx:end_idx]))
             if len(peak_indices) == 0 or (local_peak - peak_indices[-1]) > min_dist:
                 peak_indices.append(local_peak)

    stages = {
        'filtered': filt_ecg,
        'derivative': der_ecg,
        'squared': sq_ecg,
        'integrated': mwi_ecg
    }

    return np.array(peak_indices), stages


def analyze_rr_intervals(peaks: np.ndarray, fs: float = 360.0):
    if peaks is None or len(peaks) < 2:
        return {
            'num_beats': len(peaks) if peaks is not None else 0,
            'rr_intervals': np.array([]),
            'mean_rr': 0.0,
            'std_rr_sdrr': 0.0,
            'rmssd': 0.0,
            'mean_hr': 0.0,
            'min_hr': 0.0,
            'max_hr': 0.0
        }

    rr_samples = np.diff(peaks)
    rr_intervals = (rr_samples / fs) * 1000.0  # ms

    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)

    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0

    instantaneous_hr = 60000.0 / rr_intervals
    mean_hr = 60000.0 / mean_rr

    instantaneous_hr = instantaneous_hr[(instantaneous_hr >= 30) & (instantaneous_hr <= 220)]
    if len(instantaneous_hr) > 0:
        min_hr = np.min(instantaneous_hr)
        max_hr = np.max(instantaneous_hr)
    else:
        min_hr = mean_hr
        max_hr = mean_hr

    return {
        'num_beats': len(peaks),
        'rr_intervals': rr_intervals,
        'mean_rr': float(mean_rr),
        'std_rr_sdrr': float(std_rr),
        'rmssd': float(rmssd),
        'mean_hr': float(mean_hr),
        'min_hr': float(min_hr),
        'max_hr': float(max_hr)
    }


def compute_fft_spectrum(signal: np.ndarray, fs: float = 360.0):
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([]), 0.0

    detrended = signal - np.mean(signal)

    fft_vals = fft(detrended)
    fft_freqs = fftfreq(n, 1.0 / fs)

    pos_mask = fft_freqs >= 0
    freqs = fft_freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask]) / n
    magnitude[1:] = 2.0 * magnitude[1:]

    search_mask = freqs >= 0.2
    if np.sum(search_mask) > 0:
        idx = np.argmax(magnitude[search_mask])
        dominant_freq = float(freqs[search_mask][idx])
    else:
        dominant_freq = 0.0

    return freqs, magnitude, dominant_freq


def compute_psd_welch(signal: np.ndarray, fs: float = 360.0, nperseg: int = None):
    """Welch's method power spectral density estimate.

    Complements compute_fft_spectrum (a raw FFT magnitude spectrum) with a
    proper PSD estimate as required by Implementation Plan V2 section 13.
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])
    if nperseg is None:
        nperseg = min(n, int(fs * 4))
    nperseg = max(8, min(nperseg, n))
    freqs, psd = sig.welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def compute_spectral_energy_bands(freqs: np.ndarray, magnitude: np.ndarray):
    if len(freqs) == 0:
        return {'lf_power': 0.0, 'hf_power': 0.0, 'ecg_band_power': 0.0}

    power = magnitude ** 2
    total_power = np.sum(power) + 1e-12

    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    ecg_mask = (freqs >= 0.5) & (freqs <= 40.0)

    lf_power = float(np.sum(power[lf_mask]) / total_power)
    hf_power = float(np.sum(power[hf_mask]) / total_power)
    ecg_band_power = float(np.sum(power[ecg_mask]) / total_power)

    return {
        'lf_power': lf_power,
        'hf_power': hf_power,
        'ecg_band_power': ecg_band_power
    }


def compute_wavelet_scalogram(signal: np.ndarray, fs: float = 360.0, max_seconds: float = 8.0):
    try:
        import pywt
    except ImportError:
        return None

    n = min(len(signal), int(max_seconds * fs))
    x = np.asarray(signal[:n], dtype=float)
    x = x - np.mean(x)
    widths = np.arange(1, 64)
    cwtmatr, freqs = pywt.cwt(x, widths, "morl", sampling_period=1.0 / fs)
    t = np.arange(n) / fs
    return t, freqs, np.abs(cwtmatr)
