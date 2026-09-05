"""
Real-recording support: MIT-BIH Arrhythmia Database (PhysioNet).
"""

import os
import numpy as np
import pandas as pd

from src.data_loader import download_mitbih_record, load_mitbih_record
from src.preprocessing import full_dsp_pipeline
from src.feature_extraction import extract_beat_features

MITBIH_DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
MITBIH_DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]
MITBIH_FULL = MITBIH_DS1 + MITBIH_DS2

MITBIH_QUICK = ["100", "106", "119", "200", "201", "203", "208", "223"]

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}


def map_aami_label(symbol: str):
    return AAMI_MAP.get(symbol)


def download_mitbih_dataset(record_ids=None, dest_dir: str = "data/mitbih") -> list:
    record_ids = record_ids or MITBIH_QUICK
    os.makedirs(dest_dir, exist_ok=True)
    ok = []
    for rid in record_ids:
        print(f"[{rid}] downloading...")
        if download_mitbih_record(rid, dest_dir):
            ok.append(rid)
        else:
            print(f"[{rid}] FAILED - skipping")
    print(f"Downloaded {len(ok)}/{len(record_ids)} records to {dest_dir}")
    return ok


def available_mitbih_records(dest_dir: str = "data/mitbih") -> list:
    if not os.path.isdir(dest_dir):
        return []
    ids = set()
    for f in os.listdir(dest_dir):
        if f.endswith(".hea"):
            ids.add(f[:-4])
    complete = [
        rid for rid in sorted(ids)
        if all(os.path.exists(os.path.join(dest_dir, rid + ext)) for ext in (".hea", ".dat", ".atr"))
    ]
    return complete


def _select_channel(signal: np.ndarray, sig_name) -> np.ndarray:
    if signal.ndim == 1:
        return signal
    if sig_name:
        for i, name in enumerate(sig_name):
            if "MLII" in str(name).upper():
                return signal[:, i]
    return signal[:, 0]


def generate_ml_features_dataset_from_mitbih(record_ids=None, dest_dir: str = "data/mitbih") -> pd.DataFrame:
    record_ids = record_ids or available_mitbih_records(dest_dir)
    if not record_ids:
        raise FileNotFoundError(
            f"No MIT-BIH records found in '{dest_dir}'. Run "
            f"`python download_mitbih.py` first (needs internet access to physionet.org)."
        )

    all_features = []
    for rid in record_ids:
        try:
            signal, fields, ann_sample, ann_symbol = load_mitbih_record(rid, dest_dir)
        except Exception as e:
            print(f"[{rid}] could not load: {e} - skipping")
            continue
        if ann_sample is None or len(ann_sample) < 4:
            print(f"[{rid}] no usable annotations - skipping")
            continue

        sig_raw = _select_channel(signal, fields.get("sig_name"))
        fs = float(fields["fs"])
        sig_clean = full_dsp_pipeline(sig_raw, fs=fs, normalization=None)

        mapped_labels = [map_aami_label(s) for s in ann_symbol]
        keep = [i for i, lab in enumerate(mapped_labels) if lab is not None]
        if len(keep) < 4:
            continue
        peaks = ann_sample[keep]
        labels = [mapped_labels[i] for i in keep]

        record_df = extract_beat_features(sig_clean, peaks, fs=fs, labels=labels)
        if not record_df.empty:
            record_df["record_id"] = f"mitbih_{rid}"
            all_features.append(record_df)
            print(f"[{rid}] {len(record_df)} beats, labels: {pd.Series(labels).value_counts().to_dict()}")

    if not all_features:
        return pd.DataFrame()
    return pd.concat(all_features, ignore_index=True)
