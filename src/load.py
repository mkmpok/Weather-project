import pandas as pd
from src.config import DATE_COL, DATA_PATH  # keep fallback

def detect_date_column(df: pd.DataFrame):
    date_candidates = [c for c in df.columns if 'date' in c.lower()]
    return date_candidates[0] if date_candidates else DATE_COL

def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    date_col = detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df
