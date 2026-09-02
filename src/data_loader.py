import os
import numpy as np
import pandas as pd
import urllib.request

# Attempt to import wfdb (it might still be installing or already installed)
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False


def download_mitbih_record(record_id: str, destination_dir: str) -> bool:
    """
    Downloads a single MIT-BIH record from PhysioNet.
    Downloads the .hea, .dat, and .atr files.
    """
    os.makedirs(destination_dir, exist_ok=True)
    base_url = f"https://physionet.org/files/mitdb/1.0.0/{record_id}"
    extensions = [".hea", ".dat", ".atr"]
    
    success = True
    for ext in extensions:
        file_url = base_url + ext
        target_path = os.path.join(destination_dir, record_id + ext)
        
        # Skip if already exists and has size > 0
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            continue
            
        try:
            print(f"Downloading {file_url}...")
            urllib.request.urlretrieve(file_url, target_path)
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")
            success = False
            
    return success


def load_mitbih_record(record_id: str, data_dir: str):
    """
    Loads a MIT-BIH record from locally downloaded files using WFDB.
    Returns:
        signal (np.ndarray): The ECG signal (typically 2 channels).
        fields (dict): Metadata such as sampling frequency (fs), channel names, etc.
        ann_sample (np.ndarray): Indices of annotations/peaks.
        ann_symbol (list): List of annotation labels (N, V, L, R, A, etc.).
    """
    if not HAS_WFDB:
        raise ImportError("wfdb library is not installed or importable.")
        
    record_path = os.path.join(data_dir, record_id)
    
    try:
        # Load signals and header info
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        fields = {
            'fs': record.fs,
            'sig_name': record.sig_name,
            'units': record.units,
            'comments': record.comments,
            'n_sig': record.n_sig,
            'sig_len': record.sig_len
        }
    except Exception as e:
         raise IOError(f"Error loading record files for {record_id} in {data_dir}: {e}")
         
    # Try loading annotations if .atr file exists
    ann_sample = None
    ann_symbol = None
    atr_path = os.path.join(data_dir, record_id + ".atr")
    if os.path.exists(atr_path):
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            ann_sample = annotation.sample
            ann_symbol = annotation.symbol
        except Exception as e:
            print(f"Warning: could not load annotations for {record_id}: {e}")
            
    return signal, fields, ann_sample, ann_symbol


def load_csv_record(filepath: str, default_fs: float = 360.0):
    """
    Loads a CSV file containing ECG signal values.
    Accepts files with columns: (timestamp, ecg_val) or just single-column raw signal.
    Returns:
        signal (np.ndarray): 1D array of ECG signal.
        fs (float): Sampling frequency.
    Raises ValueError with a clear message on empty/malformed/non-numeric data.
    """
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Could not parse the CSV file - is it really a CSV? ({e})")

    if df.empty or len(df.columns) == 0:
        raise ValueError("The CSV file has no data rows.")

    # Try to infer sampling frequency from timestamp column if available
    time_col = None
    ecg_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower or 'sec' in col_lower or 'sample' in col_lower or col_lower == 't':
            time_col = col
        elif 'ecg' in col_lower or 'val' in col_lower or 'sig' in col_lower or 'amp' in col_lower or col_lower == 'lead':
            ecg_col = col

    if ecg_col is None:
        # Revert to the first or second column
        if len(df.columns) == 1:
            ecg_col = df.columns[0]
        else:
            ecg_col = df.columns[1] if time_col == df.columns[0] else df.columns[0]

    raw_col = pd.to_numeric(df[ecg_col], errors="coerce")
    n_nan = int(raw_col.isna().sum())
    if n_nan > 0:
        if n_nan > 0.05 * len(raw_col):
            raise ValueError(
                f"Column '{ecg_col}' has {n_nan} non-numeric value(s) out of {len(raw_col)} rows. "
                f"Check that this is really the ECG value column, not a label/annotation column."
            )
        # A handful of bad cells - interpolate rather than fail the whole file.
        raw_col = raw_col.interpolate(limit_direction="both")

    signal = raw_col.values

    fs = default_fs
    if time_col is not None and len(df) > 1:
        try:
            diffs = np.diff(df[time_col].values)
            mean_diff = np.mean(diffs)
            if mean_diff > 0:
                # If timestamp is in samples, mean_diff will be 1
                if mean_diff == 1:
                    fs = default_fs
                else:
                    fs = float(1.0 / mean_diff)
        except Exception:
            pass

    return signal, fs


def generate_gaussian_wave(t, amplitude, position, width):
    """Generates a Gaussian wave at position t with amplitude and width."""
    return amplitude * np.exp(-((t - position) ** 2) / (2 * (width ** 2)))


def generate_synthetic_ecg(duration: float = 60.0, fs: float = 360.0, 
                           heart_rate: float = 72.0, rhythm_type: str = 'normal', 
                           noise_level: float = 0.05, power_noise_level: float = 0.02,
                           baseline_noise_level: float = 0.15, seed: int = None):
    """
    Generates a highly realistic ECG signal using Gaussian wave modeling.
    Supports different rhythms:
        'normal': Normal sinus rhythm (consistent beat duration and shape)
        'bradycardia': Slow regular heart rate (HR ~ 45 BPM)
        'tachycardia': Fast regular heart rate (HR ~ 120 BPM)
        'arrhythmia': Sinus arrhythmia with random extrasystoles (PVCs, APCs, Fusion beats)
    Adds realistic noises:
        - Baseline wander (low frequency drift)
        - Power-line interference (50 Hz or 60 Hz sinusoidal noise)
        - High-frequency white noise (EMG / random noise)
    Returns:
        signal (np.ndarray): 1D array of values.
        fs (float): Sampling frequency.
        ann_sample (np.ndarray): Indices of R-peaks.
        ann_symbol (list): Annotations of beats ('N', 'V', 'S', 'F').
    """
    total_samples = int(duration * fs)
    time_axis = np.arange(total_samples) / fs
    
    # Establish beat annotations
    ann_sample = []
    ann_symbol = []
    
    # Normal beat params (times in seconds relative to R-peak)
    normal_params = {
        'P': {'amp': 0.12, 'pos': -0.18, 'wid': 0.018},
        'Q': {'amp': -0.05, 'pos': -0.07, 'wid': 0.008},
        'R': {'amp': 1.1, 'pos': 0.0, 'wid': 0.012},
        'S': {'amp': -0.28, 'pos': 0.05, 'wid': 0.012},
        'T': {'amp': 0.28, 'pos': 0.28, 'wid': 0.038}
    }
    
    # Ventricular Ectopic Beat (PVC) params (wide, no P, T inverted and huge)
    pvc_params = {
        'P': {'amp': 0.0, 'pos': -0.18, 'wid': 0.018},
        'Q': {'amp': 0.0, 'pos': -0.07, 'wid': 0.008},
        'R': {'amp': 0.7, 'pos': 0.0, 'wid': 0.035},
        'S': {'amp': -0.65, 'pos': 0.07, 'wid': 0.040},
        'T': {'amp': -0.42, 'pos': 0.28, 'wid': 0.060}
    }
    
    # Supraventricular Ectopic Beat (APC) params (normal shape, but early)
    apc_params = {
        'P': {'amp': -0.06, 'pos': -0.14, 'wid': 0.015}, # Inverted/smaller P
        'Q': {'amp': -0.05, 'pos': -0.07, 'wid': 0.008},
        'R': {'amp': 1.0, 'pos': 0.0, 'wid': 0.012},
        'S': {'amp': -0.25, 'pos': 0.05, 'wid': 0.012},
        'T': {'amp': 0.26, 'pos': 0.28, 'wid': 0.038}
    }
    
    # Fusion Beat (F) params (intermediate shape)
    fusion_params = {
        'P': {'amp': 0.05, 'pos': -0.16, 'wid': 0.018},
        'Q': {'amp': -0.02, 'pos': -0.07, 'wid': 0.008},
        'R': {'amp': 0.85, 'pos': 0.0, 'wid': 0.025},
        'S': {'amp': -0.45, 'pos': 0.06, 'wid': 0.025},
        'T': {'amp': -0.1, 'pos': 0.28, 'wid': 0.048}
    }

    # Set parameters based on the rhythm type
    target_hr = heart_rate
    if rhythm_type == 'bradycardia':
        target_hr = 45.0
    elif rhythm_type == 'tachycardia':
        target_hr = 120.0
        
    mean_rr = 60.0 / target_hr

    def _jitter_params(base_params, amp_frac=0.22, wid_frac=0.20, pos_abs=0.008):
        """Randomizes a beat's morphology slightly so beats of the same class
        aren't identical, and different classes have some natural overlap -
        without this, per-class features are a near-perfect lookup table and
        classification is trivially easy (not a real test of the model)."""
        out = {}
        for wave_name, w in base_params.items():
            out[wave_name] = {
                'amp': w['amp'] * (1.0 + np.random.normal(0, amp_frac)),
                'pos': w['pos'] + np.random.normal(0, pos_abs),
                'wid': max(0.003, w['wid'] * (1.0 + np.random.normal(0, wid_frac))),
            }
        return out
    
    # Generate beat times
    current_time = 0.35 # Padding at start
    beat_times = []
    beat_types = []
    
    # Construct sequence of beats
    if seed is not None:
        np.random.seed(seed)
    while current_time < duration - 0.5:
        # Default next interval
        rr = mean_rr
        btype = 'N'
        
        if rhythm_type == 'arrhythmia':
            # Introduce HRV variance
            rr += np.random.normal(0, 0.03 * mean_rr)
            
            # Probability-based anomalies. Multipliers are randomized (not
            # fixed) so RR-based features have realistic within-class spread
            # and some overlap between classes - a fixed multiplier per class
            # makes the label trivially readable straight off one feature.
            rand_val = np.random.rand()
            if rand_val < 0.16:  # PVC (early beat + compensatory pause)
                rr_early = mean_rr * np.clip(np.random.normal(0.65, 0.10), 0.40, 0.90)
                if current_time + rr_early < duration - 0.5:
                    beat_times.append(current_time + rr_early)
                    beat_types.append('V')
                    current_time += rr_early
                    rr = mean_rr * np.clip(np.random.normal(1.35, 0.12), 1.05, 1.65)
            elif rand_val < 0.26:  # APC
                rr_early = mean_rr * np.clip(np.random.normal(0.72, 0.10), 0.50, 0.95)
                if current_time + rr_early < duration - 0.5:
                    beat_times.append(current_time + rr_early)
                    beat_types.append('S')
                    current_time += rr_early
                    rr = mean_rr * np.clip(np.random.normal(1.15, 0.09), 0.95, 1.40)
            elif rand_val < 0.32:  # Fusion
                rr_early = mean_rr * np.clip(np.random.normal(0.80, 0.09), 0.60, 1.00)
                if current_time + rr_early < duration - 0.5:
                    beat_times.append(current_time + rr_early)
                    beat_types.append('F')
                    current_time += rr_early
                    rr = mean_rr * np.clip(np.random.normal(1.15, 0.09), 0.95, 1.40)
                    
        beat_times.append(current_time + rr)
        beat_types.append(btype)
        current_time += rr
        
    # Generate ECG signal
    ecg_signal = np.zeros(total_samples)
    
    for b_time, b_type in zip(beat_times, beat_types):
        # Peak index in sample index
        peak_idx = int(b_time * fs)
        if peak_idx >= total_samples:
            continue
            
        ann_sample.append(peak_idx)
        ann_symbol.append(b_type)
        
        # Pick parameters based on beat type
        if b_type == 'N':
            params = normal_params
        elif b_type == 'V':
            params = pvc_params
        elif b_type == 'S':
            params = apc_params
        elif b_type == 'F':
            params = fusion_params
        else:
            params = normal_params
        params = _jitter_params(params)
            
        # Draw the waves
        # Determine the window around R-peak (e.g. -0.4s to +0.5s)
        start_t = max(0.0, b_time - 0.4)
        end_t = min(duration, b_time + 0.6)
        
        start_idx = int(start_t * fs)
        end_idx = int(end_t * fs)
        
        for idx in range(start_idx, end_idx):
            t = (idx / fs) - b_time
            val = 0.0
            for name, wave in params.items():
                val += generate_gaussian_wave(t, wave['amp'], wave['pos'], wave['wid'])
            ecg_signal[idx] += val

    # Noise 1: Baseline Wander (drift due to respiration ~0.15 Hz to 0.3 Hz)
    # Modeled as sum of slow sines + random walks
    slow_drift = (
        baseline_noise_level * 0.6 * np.sin(2 * np.pi * 0.15 * time_axis) +
        baseline_noise_level * 0.4 * np.sin(2 * np.pi * 0.3 * time_axis) +
        baseline_noise_level * 0.3 * np.cos(2 * np.pi * 0.05 * time_axis)
    )
    ecg_signal += slow_drift
    
    # Noise 2: Power-line interference (50 Hz or 60 Hz sinusoidal noise)
    power_noise = power_noise_level * np.sin(2 * np.pi * 50.0 * time_axis)
    ecg_signal += power_noise
    
    # Noise 3: High-frequency EMG/muscle noise
    hf_noise = np.random.normal(0, noise_level, total_samples)
    ecg_signal += hf_noise
    
    return ecg_signal, fs, np.array(ann_sample), ann_symbol