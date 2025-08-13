import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.config import CITY_COL, DATE_COL

def detect_anomalies_city(df: pd.DataFrame, city: str, value_col: str, contamination=0.01):
    cdf = df[df[CITY_COL]==city].dropna(subset=[value_col]).sort_values(DATE_COL).copy()
    if cdf.empty:
        return cdf, pd.Series(dtype=int)
    X = cdf[[value_col]].values
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)  # -1 anomaly, 1 normal
    cdf["anomaly"] = (labels == -1).astype(int)
    return cdf, cdf["anomaly"]