import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/history.db"

def init_database():
    """Initializes the SQLite database and creates the necessary tables.

    Schema per Implementation Plan V2 section 41:
      - analyses: one row per completed analysis run (adds patient_id,
        predicted_class, confidence, model_version vs. the original schema).
      - analysis_features: per-analysis feature name/value pairs.
      - model_runs: optional log of individual training runs.
    Uses ALTER TABLE for any columns missing from a pre-existing DB file so
    upgrading an older history.db does not lose existing rows.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_name TEXT NOT NULL,
        patient_id TEXT,
        timestamp TEXT NOT NULL,
        sampling_frequency REAL NOT NULL,
        duration REAL NOT NULL,
        signal_quality TEXT NOT NULL,
        heart_rate REAL NOT NULL,
        predicted_class TEXT,
        prediction TEXT NOT NULL,
        confidence REAL,
        model_name TEXT NOT NULL,
        model_version TEXT,
        report_path TEXT
    )
    """)

    # Backfill columns for DBs created by an older version of this schema.
    cursor.execute("PRAGMA table_info(analyses)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, coltype in [
        ("patient_id", "TEXT"),
        ("predicted_class", "TEXT"),
        ("confidence", "REAL"),
        ("model_version", "TEXT"),
    ]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE analyses ADD COLUMN {col} {coltype}")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER NOT NULL,
        feature_name TEXT NOT NULL,
        feature_value REAL,
        FOREIGN KEY (analysis_id) REFERENCES analyses (id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_version TEXT,
        dataset TEXT,
        accuracy REAL,
        macro_f1 REAL,
        weighted_f1 REAL,
        roc_auc REAL,
        training_timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_analysis_history(record_name: str, fs: float, duration: float,
                          signal_quality: str, heart_rate: float,
                          prediction: str, model_name: str, report_path: str = None,
                          patient_id: str = None, predicted_class: str = None,
                          confidence: float = None, model_version: str = None,
                          features: dict = None):
    """
    Saves a completed ECG analysis run to the database, and optionally a
    per-feature breakdown into analysis_features (used for later inspection
    and to satisfy Implementation Plan V2 section 41's feature-storage table).
    """
    init_database()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO analyses (record_name, patient_id, timestamp, sampling_frequency, duration,
                          signal_quality, heart_rate, predicted_class, prediction, confidence,
                          model_name, model_version, report_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (record_name, patient_id, timestamp, fs, duration, signal_quality, heart_rate,
          predicted_class or prediction, prediction, confidence, model_name, model_version, report_path))

    analysis_id = cursor.lastrowid

    if features:
        for f_name, f_val in features.items():
            try:
                f_val = float(f_val)
            except (TypeError, ValueError):
                continue
            cursor.execute(
                "INSERT INTO analysis_features (analysis_id, feature_name, feature_value) VALUES (?, ?, ?)",
                (analysis_id, f_name, f_val),
            )

    conn.commit()
    conn.close()
    print(f"Analysis saved to database for record: {record_name}")


def get_analysis_history() -> pd.DataFrame:
    init_database()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM analyses ORDER BY id DESC", conn)
    except Exception as e:
        print(f"Error loading database history: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def get_analysis_features(analysis_id: int) -> pd.DataFrame:
    """Returns the stored feature name/value pairs for one analysis row."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT feature_name, feature_value FROM analysis_features WHERE analysis_id = ?",
            conn, params=(analysis_id,),
        )
    except Exception as e:
        print(f"Error loading analysis features: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def save_model_run(model_name: str, dataset: str, accuracy: float, macro_f1: float,
                    weighted_f1: float, roc_auc: float, model_version: str = None):
    """Optional log of a single training run's headline metrics (plan section 41)."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO model_runs (model_name, model_version, dataset, accuracy, macro_f1, weighted_f1, roc_auc, training_timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (model_name, model_version, dataset, accuracy, macro_f1, weighted_f1, roc_auc, timestamp))
    conn.commit()
    conn.close()


def clear_analysis_history():
    init_database()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM analyses")
        cursor.execute("DELETE FROM analysis_features")
        conn.commit()
    except Exception as e:
        print(f"Error clearing database: {e}")
    finally:
        conn.close()
