import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq

def pan_tompkins_detector(data: np.ndarray, fs: float = 360.0):
    """
    Implements the Pan-Tompkins QRS detection algorithm:
    1. Band-pass Filter (5-15 Hz)
    2. Differentiation
    3. Squaring
    4. Moving Window Integration (approx. 150ms window)
    5. Adaptive thresholding to identify R-peaks.
    
    Returns:
        peaks (np.ndarray): Indices of detected R-peaks.
        stages (dict): Intermediate signals for visualization.
    """
    # 1. Bandpass filter 5-15 Hz to isolate QRS energy
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = sig.butter(3, [low, high], btype='band')
    filt_ecg = sig.filtfilt(b, a, data)
    
    # 2. Derivative (approximate slope of QRS)
    # y(n) = 1/8 [2x(n) + x(n-1) - x(n-3) - 2x(n-4)]
    # We can use standard np.diff or a five-point derivative filter:
    der_ecg = np.zeros_like(filt_ecg)
    der_ecg[4:] = (
        2.0 * filt_ecg[4:]
        + filt_ecg[3:-1]
        - filt_ecg[1:-3]
        - 2.0 * filt_ecg[:-4]
    ) / 8.0
        
    # 3. Squaring
    sq_ecg = der_ecg ** 2
    
    # 4. Moving Window Integration
    # Integration window of ~150 ms
    win_samples = int(0.150 * fs)
    if win_samples % 2 == 0:
        win_samples += 1
    
    # Convolve with moving window filter
    integration_kernel = np.ones(win_samples) / win_samples
    mwi_ecg = np.convolve(sq_ecg, integration_kernel, mode='same')
    
    # 5. Peak Detection & Adaptive Thresholds
    # Initialize thresholds
    peak_indices = []
    
    # Find local maxima in baseline/unprocessed signal or integrated signal
    # First, run a standard find peaks on the MWI signal to get candidates
    min_dist = int(0.250 * fs) # Refractory period of 250 ms
    # Peak threshold starts based on amplitude
    height_threshold = np.percentile(mwi_ecg, 90) * 0.3
    
    candidates, _ = sig.find_peaks(mwi_ecg, distance=min_dist, height=height_threshold)
    
    # Map candidates back to the actual peaks in the original signal
    search_window = int(0.100 * fs) # Check within +/- 100ms in original signal
    for cand in candidates:
        start_idx = max(0, cand - search_window)
        end_idx = min(len(data), cand + search_window)
        if start_idx < end_idx:
             # R-peak corresponds to local maximum (absolute value) in the preprocessed ECG
             local_peak = start_idx + np.argmax(np.abs(data[start_idx:end_idx]))
             # Ensure the peak isn't already added (due to overlap)
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
    """
    Computes time-domain metrics for R-peaks and RR intervals.
    Returns:
        metrics (dict): Standard HRV and beat metrics:
            - num_beats: Total count of peaks
            - rr_intervals: Array of RR intervals in milliseconds
            - mean_rr: Mean RR interval (ms)
            - std_rr_sdrr: Standard deviation of RR intervals (SDRR) (ms)
            - rmssd: Root mean square of successive differences (ms)
            - mean_hr: Mean heart rate (BPM)
            - min_hr: Minimum instantaneous heart rate (BPM)
            - max_hr: Maximum instantaneous heart rate (BPM)
    """
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
        
    # RR intervals in samples -> convert to seconds, then milliseconds
    rr_samples = np.diff(peaks)
    rr_intervals = (rr_samples / fs) * 1000.0  # ms
    
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    
    # Successive differences
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0
    
    # Heart rates (BPM)
    # Instantaneous Heart Rate: 60 / (rr_interval in seconds) = 60000 / (rr_interval in ms)
    instantaneous_hr = 60000.0 / rr_intervals
    mean_hr = 60000.0 / mean_rr
    
    # Handle possible infinite/extreme heart rates due to noise peaks
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
    """
    Computes Fast Fourier Transform of the ECG signal.
    Returns:
        freqs (np.ndarray): Positive frequencies axis.
        magnitude (np.ndarray): Magnitude spectrum (single-sided).
        dominant_freq (float): Main frequency component (excluding near-DC < 0.2 Hz).
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([]), 0.0
        
    # Remove DC offset (mean)
    detrended = signal - np.mean(signal)
    
    # Compute FFT
    fft_vals = fft(detrended)
    fft_freqs = fftfreq(n, 1.0 / fs)
    
    # Get physical positive frequencies
    pos_mask = fft_freqs >= 0
    freqs = fft_freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask]) / n
    # Since it is single-sided, double the magnitude for positive frequencies
    magnitude[1:] = 2.0 * magnitude[1:]
    
    # Identify dominant frequency
    # Filter out near-DC frequencies below 0.2 Hz to avoid residual breathing artifacts
    search_mask = freqs >= 0.2
    if np.sum(search_mask) > 0:
        idx = np.argmax(magnitude[search_mask])
        dominant_freq = float(freqs[search_mask][idx])
    else:
        dominant_freq = 0.0
        
    return freqs, magnitude, dominant_freq


def compute_spectral_energy_bands(freqs: np.ndarray, magnitude: np.ndarray):
    """
    Calculates proportion of signal energy in standard frequency bands:
        - Low Frequency (LF): 0.04 - 0.15 Hz (Sympathetic/Parasympathetic balance index)
        - High Frequency (HF): 0.15 - 0.4 Hz (Parasympathetic activity index)
        - VHF / ECG Frequency: 0.5 - 40 Hz (Main QRS energy band)
    """
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
    """
    Optional CWT (Morlet) of a short ECG window. ECG is non-stationary; wavelets
    show transient QRS energy across scales. Returns time, frequencies, |CWT|.
    """
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
