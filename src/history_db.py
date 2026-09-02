import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/history.db"

def init_database():
    """Initializes the SQLite database and creates the necessary tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create analysis history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        sampling_frequency REAL NOT NULL,
        duration REAL NOT NULL,
        signal_quality TEXT NOT NULL,
        heart_rate REAL NOT NULL,
        prediction TEXT NOT NULL,
        model_name TEXT NOT NULL,
        report_path TEXT
    )
    """)
    
    conn.commit()
    conn.close()


def save_analysis_history(record_name: str, fs: float, duration: float, 
                          signal_quality: str, heart_rate: float, 
                          prediction: str, model_name: str, report_path: str = None):
    """
    Saves a completed ECG analysis run in the database historical logs log.
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
    INSERT INTO analyses (record_name, timestamp, sampling_frequency, duration, 
                          signal_quality, heart_rate, prediction, model_name, report_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (record_name, timestamp, fs, duration, signal_quality, heart_rate, prediction, model_name, report_path))
    
    conn.commit()
    conn.close()
    print(f"Analysis saved to database for record: {record_name}")


def get_analysis_history() -> pd.DataFrame:
    """
    Retrieves all records from the historical logs table as a pandas DataFrame.
    """
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


def clear_analysis_history():
    """Clears all historical analysis logs from the SQLite database."""
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM analyses")
        conn.commit()
    except Exception as e:
        print(f"Error clearing database: {e}")
    finally:
        conn.close()
