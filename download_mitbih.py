"""
Download real ECG recordings from the MIT-BIH Arrhythmia Database (PhysioNet)
so the models can be trained on real patient data instead of only the
synthetic generator.

Usage:
    python download_mitbih.py            # quick set, 8 records, ~15 MB
    python download_mitbih.py --full      # full AAMI set, 44 records, ~100 MB
    python download_mitbih.py --records 100 101 208   # specific records

Requires internet access to physionet.org. Files are saved to data/mitbih/
and are safe to re-run (already-downloaded files are skipped).
"""

import argparse
import sys

from src.mitbih_data import MITBIH_QUICK, MITBIH_FULL, download_mitbih_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia Database records.")
    parser.add_argument("--full", action="store_true", help="Download the full 44-record AAMI set instead of the quick 8-record subset.")
    parser.add_argument("--records", nargs="+", default=None, help="Specific record IDs to download, e.g. --records 100 101 208")
    parser.add_argument("--dest", default="data/mitbih", help="Destination folder (default: data/mitbih)")
    args = parser.parse_args()

    if args.records:
        record_ids = args.records
    elif args.full:
        record_ids = MITBIH_FULL
    else:
        record_ids = MITBIH_QUICK

    print(f"Downloading {len(record_ids)} record(s) to {args.dest}/ ...")
    ok = download_mitbih_dataset(record_ids, dest_dir=args.dest)

    if len(ok) < len(record_ids):
        print(f"\nWarning: {len(record_ids) - len(ok)} record(s) failed to download.", file=sys.stderr)
    if not ok:
        sys.exit(1)

    print(f"\nDone. Next step:  python -m src.ml_training --source both")
