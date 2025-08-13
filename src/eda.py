import pandas as pd
import matplotlib.pyplot as plt
from src.config import DATE_COL, TEMP_COL, PRECIP_COL, CITY_COL
from src.utils import safe_savefig

def plot_time_series_city(df: pd.DataFrame, city: str, value_col: str, out_path: str, title: str):
    cdf = df[df[CITY_COL] == city].dropna(subset=[value_col]).sort_values(DATE_COL)
    if cdf.empty:
        return False
    plt.figure(figsize=(10,4))
    plt.plot(cdf[DATE_COL], cdf[value_col])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(value_col)
    safe_savefig(out_path)
    return True

def plot_correlation_heatmap(df: pd.DataFrame, out_path: str):
    num_df = df.select_dtypes("number")
    if num_df.shape[1] < 2:
        return False
    corr = num_df.corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, aspect='auto')
    plt.colorbar(label="Correlation")
    plt.title("Correlation (numeric features)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    safe_savefig(out_path)
    return True

def seasonal_monthly(df: pd.DataFrame, city: str, value_col: str, out_path: str):
    cdf = df[df[CITY_COL]==city].dropna(subset=[value_col]).copy()
    if cdf.empty: return False
    cdf["month"] = cdf[DATE_COL].dt.month
    agg = cdf.groupby("month")[value_col].mean()
    plt.figure(figsize=(8,4))
    plt.plot(agg.index, agg.values, marker="o")
    plt.title(f"Monthly Seasonal Mean — {city} — {value_col}")
    plt.xlabel("Month"); plt.ylabel(value_col)
    safe_savefig(out_path)
    return True