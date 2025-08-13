import numpy as np
import pandas as pd
from src.config import DATE_COL

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        # Clip to 1st–99th percentile (robust to extreme outliers)
        lo, hi = out[col].quantile([0.01, 0.99])
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].interpolate(limit_direction="both")
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out

def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        mu, sigma = out[col].mean(), out[col].std()
        if sigma > 0:
            out[col] = (out[col] - mu) / sigma
    return out

def add_time_parts(out, date_col=None):
    if date_col is None or date_col not in out.columns:
        # Try to auto-detect column name containing 'date' or 'updated'
        possible_cols = [col for col in out.columns if 'date' in col.lower() or 'update' in col.lower()]
        if possible_cols:
            date_col = possible_cols[0]  # pick the first match
            print(f"[INFO] Detected date column: {date_col}")
        else:
            raise ValueError("❌ No date column found in dataset!")

    # Convert to datetime
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["hour"] = out[date_col].dt.hour
    out["minute"] = out[date_col].dt.minute
    return out

def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = df.drop_duplicates()
    df = handle_outliers(df)
    df = impute_missing(df)
    df = add_time_parts(df)
    return df