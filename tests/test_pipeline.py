import os
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_ecg, load_csv_record
from src.signal_quality import assess_signal_quality
from src.preprocessing import full_dsp_pipeline, validate_ecg_signal
from src.signal_analysis import pan_tompkins_detector, analyze_rr_intervals, compute_fft_spectrum
from src.feature_extraction import extract_beat_features
from src.history_db import save_analysis_history, get_analysis_history, clear_analysis_history
from src.pipeline import run_full_analysis
from src.mitbih_data import map_aami_label, AAMI_MAP


def test_synthetic_generation():
    duration = 10.0
    fs = 360.0
    sig, fs_out, peaks, symbols = generate_synthetic_ecg(
         duration=duration, fs=fs, rhythm_type='normal', seed=42
    )
    assert len(sig) == int(duration * fs)
    assert fs_out == fs
    assert len(peaks) > 0
    assert len(peaks) == len(symbols)
    assert isinstance(sig, np.ndarray)


def test_signal_quality_metrics():
    fs = 360.0
    sig, _, _, _ = generate_synthetic_ecg(duration=5.0, fs=fs, rhythm_type='normal', seed=42)
    metrics = assess_signal_quality(sig, fs=fs)

    assert 'status' in metrics
    assert 'details' in metrics
    assert metrics['status'] in ['GOOD', 'ACCEPTABLE', 'POOR']
    assert 'sqi_score' in metrics
    assert 0 <= metrics['sqi_score'] <= 100

    flat_sig = np.zeros(1000)
    flat_metrics = assess_signal_quality(flat_sig, fs=fs)
    assert flat_metrics['status'] == 'POOR'
    assert 'flatlin' in flat_metrics['details'].lower()
    assert flat_metrics['sqi_score'] <= 50


def test_filtering_pipeline():
    duration = 5.0
    fs = 360.0
    sig, _, _, _ = generate_synthetic_ecg(duration=duration, fs=fs, rhythm_type='normal', seed=42)

    sig_filtered = full_dsp_pipeline(
         sig, fs=fs,
         use_baseline=True, baseline_method='butterworth', baseline_cutoff=0.5,
         use_notch=True, notch_freq=50.0,
         use_bandpass=True, bp_low=0.5, bp_high=40.0, bp_order=4,
         normalization='z-score'
    )

    assert len(sig_filtered) == len(sig)
    assert isinstance(sig_filtered, np.ndarray)
    assert np.abs(np.mean(sig_filtered)) < 0.1
    assert np.abs(np.std(sig_filtered) - 1.0) < 0.1


def test_pan_tompkins_peak_detection():
    duration = 10.0
    fs = 360.0
    sig, _, gt_peaks, _ = generate_synthetic_ecg(duration=duration, fs=fs, rhythm_type='normal', seed=42)
    sig_clean = full_dsp_pipeline(sig, fs=fs)

    detected_peaks, pt_stages = pan_tompkins_detector(sig_clean, fs=fs)

    assert len(detected_peaks) > 0
    assert 'filtered' in pt_stages
    assert 'integrated' in pt_stages
    assert len(detected_peaks) == pytest.approx(len(gt_peaks), abs=3)


def test_rr_variability_analytics():
    peaks = np.array([360, 720, 1080, 1440, 1800])
    fs = 360.0

    metrics = analyze_rr_intervals(peaks, fs=fs)

    assert metrics['num_beats'] == len(peaks)
    assert metrics['mean_rr'] == pytest.approx(1000.0, abs=1.0)
    assert metrics['mean_hr'] == pytest.approx(60.0, abs=1.0)
    assert metrics['std_rr_sdrr'] == pytest.approx(0.0, abs=1.0)
    assert metrics['rmssd'] == pytest.approx(0.0, abs=1.0)


def test_features_matrix_extraction():
    duration = 10.0
    fs = 360.0
    sig, _, peaks, _ = generate_synthetic_ecg(duration=duration, fs=fs, rhythm_type='normal', seed=42)
    sig_clean = full_dsp_pipeline(sig, fs=fs)

    feat_df = extract_beat_features(sig_clean, peaks, fs=fs)

    assert isinstance(feat_df, pd.DataFrame)
    assert not feat_df.empty
    expected_cols = ['pre_rr', 'post_rr', 'rr_ratio', 'qrs_amplitude', 'qrs_duration_est', 'dominant_frequency']
    for col in expected_cols:
         assert col in feat_df.columns


def test_database_logging():
    clear_analysis_history()

    save_analysis_history(
         record_name="TST_999", fs=360.0, duration=60.0, signal_quality="GOOD",
         heart_rate=72.0, prediction="Normal Sinus Rhythm", model_name="Random Forest",
         report_path="reports/dummy.pdf",
    )

    df = get_analysis_history()
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]['record_name'] == "TST_999"
    assert df.iloc[0]['signal_quality'] == "GOOD"

    clear_analysis_history()
    df_empty = get_analysis_history()
    assert df_empty.empty or len(df_empty) == 0


def test_database_stores_confidence_and_model_version():
    """Regression test for the schema fix - confidence/model_version/predicted_class
    must actually persist (Implementation Plan V2 §41)."""
    clear_analysis_history()
    save_analysis_history(
        record_name="TST_CONF", fs=360.0, duration=30.0, signal_quality="GOOD",
        heart_rate=80.0, prediction="Normal Sinus Beat (N)", model_name="Random Forest",
        report_path=None, patient_id="P001", predicted_class="N",
        confidence=0.87, model_version="v1.0", features={"mean": 0.1, "std": 0.2},
    )
    df = get_analysis_history()
    row = df.iloc[0]
    assert row['patient_id'] == "P001"
    assert row['predicted_class'] == "N"
    assert row['confidence'] == pytest.approx(0.87)
    assert row['model_version'] == "v1.0"
    clear_analysis_history()


def test_validate_rejects_empty_signal():
    with pytest.raises(ValueError, match="empty"):
        validate_ecg_signal(np.array([]), fs=360.0)


def test_validate_rejects_too_short_signal():
    with pytest.raises(ValueError, match="too short"):
        validate_ecg_signal(np.random.randn(50), fs=360.0)


def test_validate_rejects_nan_and_inf():
    sig = np.random.randn(2000)
    sig[10] = np.nan
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_ecg_signal(sig, fs=360.0)

    sig2 = np.random.randn(2000)
    sig2[10] = np.inf
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_ecg_signal(sig2, fs=360.0)


def test_validate_rejects_flat_signal():
    with pytest.raises(ValueError, match="flat"):
        validate_ecg_signal(np.ones(2000) * 0.5, fs=360.0)


def test_validate_rejects_bad_fs():
    with pytest.raises(ValueError, match="Invalid sampling frequency"):
        validate_ecg_signal(np.random.randn(2000), fs=0)
    with pytest.raises(ValueError, match="Invalid sampling frequency"):
        validate_ecg_signal(np.random.randn(2000), fs=-360.0)


def test_validate_accepts_multichannel_by_taking_first_column():
    fs = 360.0
    sig, _, _, _ = generate_synthetic_ecg(duration=5.0, fs=fs, rhythm_type='normal', seed=42)
    two_channel = np.stack([sig, sig * 0.5], axis=1)
    cleaned = validate_ecg_signal(two_channel, fs=fs)
    assert cleaned.ndim == 1
    assert len(cleaned) == len(sig)


def test_run_full_analysis_raises_clean_error_on_bad_input():
    with pytest.raises(ValueError):
        run_full_analysis(np.random.randn(20), fs=360.0)


def test_run_full_analysis_end_to_end_ok():
    fs = 360.0
    sig, _, _, _ = generate_synthetic_ecg(duration=15.0, fs=fs, rhythm_type='arrhythmia', seed=7)
    result = run_full_analysis(sig, fs=fs, run_prediction=False)
    assert result["sig_raw"] is not None
    assert result["sig_clean"] is not None
    assert len(result["peaks"]) > 0
    assert "status" in result["quality"]


def test_run_full_analysis_can_skip_prediction():
    """DSP-page performance fix: run_prediction=False must not touch the model files."""
    fs = 360.0
    sig, _, _, _ = generate_synthetic_ecg(duration=10.0, fs=fs, rhythm_type='normal', seed=7)
    result = run_full_analysis(sig, fs=fs, run_prediction=False)
    assert result["prediction_ran"] is False
    assert result["pred_df"] is None
    assert result["dominant_class"] is None


def test_load_csv_record_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_csv_record(str(p))


def test_load_csv_record_rejects_mostly_non_numeric(tmp_path):
    p = tmp_path / "bad.csv"
    df = pd.DataFrame({"time": range(20), "ecg": ["x"] * 20})
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="non-numeric"):
        load_csv_record(str(p))


def test_load_csv_record_interpolates_a_few_bad_cells(tmp_path):
    p = tmp_path / "mostly_ok.csv"
    vals = list(np.random.randn(200).astype(str))
    vals[5] = "not_a_number"
    df = pd.DataFrame({"time": range(200), "ecg": vals})
    df.to_csv(p, index=False)
    signal, fs = load_csv_record(str(p))
    assert len(signal) == 200
    assert not np.isnan(signal).any()


def test_load_csv_record_valid_file(tmp_path):
    p = tmp_path / "good.csv"
    fs = 360.0
    sig, _, _, _ = generate_synthetic_ecg(duration=3.0, fs=fs, rhythm_type='normal', seed=1)
    df = pd.DataFrame({"time": np.arange(len(sig)) / fs, "ecg": sig})
    df.to_csv(p, index=False)
    signal, fs_out = load_csv_record(str(p), default_fs=fs)
    assert len(signal) == len(sig)
    assert fs_out == pytest.approx(fs, rel=0.05)


def test_aami_label_mapping_known_symbols():
    assert map_aami_label("N") == "N"
    assert map_aami_label("L") == "N"
    assert map_aami_label("V") == "V"
    assert map_aami_label("E") == "V"
    assert map_aami_label("A") == "S"
    assert map_aami_label("F") == "F"


def test_aami_label_mapping_drops_unknown_symbols():
    assert map_aami_label("/") is None
    assert map_aami_label("Q") is None
    assert map_aami_label("+") is None


def test_aami_map_only_contains_four_target_classes():
    assert set(AAMI_MAP.values()) == {"N", "S", "V", "F"}
