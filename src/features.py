import pandas as pd
from src.config import DATE_COL

def add_rolling_lags(ts: pd.DataFrame, target: str, windows=(7,14,30)):
    ts = ts.sort_values(DATE_COL).copy()
    for w in windows:
        ts[f"{target}ma{w}"] = ts[target].rolling(w, min_periods=1).mean()
        ts[f"{target}lag{w}"] = ts[target].shift(w)
    ts = ts.dropna()
    return ts