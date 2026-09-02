import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def extract_beat_features(signal: np.ndarray, peaks: np.ndarray, fs: float = 360.0, labels: list = None):
    """
    Extracts numerical features for each detected heartbeat peak.
    Returns:
        features_df (pd.DataFrame): Tabular feature vectors for machine learning training.
    """
    num_peaks = len(peaks)
    if num_peaks == 0:
        return pd.DataFrame()
        
    features = []
    
    # Calculate global average RR for boundary padding
    if num_peaks > 1:
        rr_intervals = np.diff(peaks) / fs  # in seconds
        avg_rr = np.mean(rr_intervals)
    else:
        avg_rr = 0.8  # Default ~75 BPM
        
    # Segment window parameters relative to R-peak
    # 250ms window: -90ms to +160ms (at 360Hz: -32 samples to +58 samples)
    pre_samples = int(0.09 * fs)
    post_samples = int(0.16 * fs)
    
    for i in range(num_peaks):
        peak_idx = peaks[i]
        
        # 1. Temporal / RR Features
        # Preceding RR interval
        if i == 0:
             pre_rr = avg_rr
        else:
             pre_rr = (peaks[i] - peaks[i-1]) / fs
             
        # Succeeding RR interval
        if i == num_peaks - 1:
             post_rr = avg_rr
        else:
             post_rr = (peaks[i+1] - peaks[i]) / fs
             
        rr_ratio = pre_rr / (post_rr + 1e-12)
        
        # Local mean RR (average of up to 10 surrounding beats)
        start_idx = max(0, i-5)
        end_idx = min(num_peaks, i+6)
        if start_idx < end_idx - 1:
             surrounding_peaks = peaks[start_idx:end_idx]
             local_rr_mean = np.mean(np.diff(surrounding_peaks)) / fs
        else:
             local_rr_mean = avg_rr
             
        # 2. QRS Window Segmentation
        seg_start = peak_idx - pre_samples
        seg_end = peak_idx + post_samples
        
        # Check boundary integrity
        if seg_start < 0 or seg_end > len(signal):
             # Zero-pad or crop if peak is too close to signal boundaries
             left_pad = max(0, -seg_start)
             right_pad = max(0, seg_end - len(signal))
             
             slice_start = max(0, seg_start)
             slice_end = min(len(signal), seg_end)
             segment = np.concatenate([
                 np.zeros(left_pad),
                 signal[slice_start:slice_end],
                 np.zeros(right_pad)
             ])
        else:
             segment = signal[seg_start:seg_end]
             
        # 3. Statistical Features
        seg_mean = np.mean(segment)
        seg_var = np.var(segment)
        seg_std = np.std(segment)
        seg_skew = skew(segment)
        seg_kurt = kurtosis(segment)
        seg_energy = np.sum(segment ** 2)
        
        # 4. Morphological Features
        r_val = signal[peak_idx]
        
        # Find Q-wave (minimum in 50ms window before R-peak)
        q_window = int(0.05 * fs)
        q_start = max(0, peak_idx - q_window)
        q_val = np.min(signal[q_start:peak_idx+1]) if q_start < peak_idx else r_val
        
        # Find S-wave (minimum in 60ms window after R-peak)
        s_window = int(0.06 * fs)
        s_end = min(len(signal), peak_idx + s_window)
        s_val = np.min(signal[peak_idx:s_end+1]) if peak_idx < s_end else r_val
        
        qrs_amplitude = r_val - min(q_val, s_val)
        qs_width_samples = (peak_idx - q_start) + (s_end - peak_idx) # Quick QRS duration proxy
        qrs_duration_est = qs_width_samples / fs
        
        # 5. Spectral Features of the segment (FFT)
        n_fft = len(segment)
        seg_fft = np.abs(fft(segment - seg_mean)) / n_fft
        seg_fft = seg_fft[:n_fft//2] # Single sided
        
        dominant_idx = np.argmax(seg_fft) if len(seg_fft) > 0 else 0
        freq_step = fs / n_fft
        dominant_freq = dominant_idx * freq_step
        
        # Spectral entropy (complexity)
        prob = (seg_fft ** 2) / (np.sum(seg_fft ** 2) + 1e-12)
        prob = prob[prob > 0]
        spectral_entropy = -np.sum(prob * np.log2(prob)) if len(prob) > 0 else 0.0
        
        # Package feature dict
        beat_features = {
            'beat_index': int(peak_idx),
            'pre_rr': float(pre_rr),
            'post_rr': float(post_rr),
            'rr_ratio': float(rr_ratio),
            'local_rr_mean': float(local_rr_mean),
            'mean': float(seg_mean),
            'variance': float(seg_var),
            'std': float(seg_std),
            'skewness': float(seg_skew),
            'kurtosis': float(seg_kurt),
            'energy': float(seg_energy),
            'r_height': float(r_val),
            'q_depth': float(q_val),
            's_depth': float(s_val),
            'qrs_amplitude': float(qrs_amplitude),
            'qrs_duration_est': float(qrs_duration_est),
            'dominant_frequency': float(dominant_freq),
            'spectral_entropy': float(spectral_entropy)
        }
        
        # Attach ground truth labels during training
        if labels is not None and i < len(labels):
            beat_features['label'] = labels[i]
            
        features.append(beat_features)
        
    return pd.DataFrame(features)
