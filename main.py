import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DATA_PATH, REPORTS_DIR, FIG_DIR, REPORT_MD,
    DATE_COL, CITY_COL, TEMP_COL, PRECIP_COL,
    TEST_SIZE_FRACTION, ROLL_WINDOWS
)
from src.utils import ensure_dirs, safe_savefig
from src.load import load_raw
from src.preprocess import preprocess_all
from src.eda import plot_time_series_city, plot_correlation_heatmap, seasonal_monthly
from src.anomalies import detect_anomalies_city
from src.features import add_rolling_lags
from src.models import (
    fit_sarimax, predict_sarimax, fit_xgb, fit_lgbm,
    metrics, get_permutation_importance, ensemble_preds
)
from src.spatial import average_by_country, choropleth_if_available
from src.report import write_report

def pick_city(df: pd.DataFrame) -> str:
    vc = df[CITY_COL].dropna().value_counts()
    return vc.index[0] if not vc.empty else None

def train_test_split_by_time(df: pd.DataFrame, frac=TEST_SIZE_FRACTION):
    n = len(df)
    split = int(n * (1 - frac))
    return df.iloc[:split], df.iloc[split:]

def main():
    ensure_dirs(REPORTS_DIR, FIG_DIR)

    # 1) Load & preprocess
    df_raw = load_raw(DATA_PATH)
    df = preprocess_all(df_raw)

    # Columns may have been standardized to lowercase; ensure config matches
    # Adjust if necessary:
    cols = set(df.columns)
    temp_col = TEMP_COL if TEMP_COL in cols else ("temperature_c" if "temperature_c" in cols else None)
    precip_col = PRECIP_COL if PRECIP_COL in cols else ("precipitation" if "precipitation" in cols else None)
    if temp_col is None:
        print("Could not find a temperature column. Update src/config.py to match your CSV.")
        return

    # 2) EDA (Basic)
    city = pick_city(df)
    eda_figs = []

    ok = plot_time_series_city(df, city, temp_col, FIG_DIR / "temp_time.png",
                               f"{city} — {temp_col} over time")
    if ok: eda_figs.append(FIG_DIR / "temp_time.png")

    if precip_col:
        ok = plot_time_series_city(df, city, precip_col, FIG_DIR / "precip_time.png",
                                   f"{city} — {precip_col} over time")
        if ok: eda_figs.append(FIG_DIR / "precip_time.png")

    ok = plot_correlation_heatmap(df, FIG_DIR / "correlations.png")
    if ok: eda_figs.append(FIG_DIR / "correlations.png")

    ok = seasonal_monthly(df, city, temp_col, FIG_DIR / "seasonal_temp.png")
    if ok: eda_figs.append(FIG_DIR / "seasonal_temp.png")

    # 3) Advanced EDA — Anomaly Detection
    anomalies_plot = None
    cdf, labels = detect_anomalies_city(df, city, temp_col, contamination=0.01)
    if not cdf.empty:
        plt.figure(figsize=(10,4))
        plt.plot(cdf[DATE_COL], cdf[temp_col], label="value")
        plt.scatter(cdf.loc[cdf["anomaly"]==1, DATE_COL],
                    cdf.loc[cdf["anomaly"]==1, temp_col], marker="x")
        plt.title(f"Anomalies — {city} — {temp_col}")
        plt.legend()
        anomalies_plot = FIG_DIR / "anomalies.png"
        safe_savefig(anomalies_plot.as_posix())

    # 4) Forecasting with multiple models
    city_df = df[df[CITY_COL]==city].dropna(subset=[temp_col]).sort_values(DATE_COL).copy()
    if city_df.empty:
        print("No rows for selected city/target.")
        return

    # SARIMAX baseline
    y = city_df[temp_col].reset_index(drop=True)
    split = int(len(y) * (1 - TEST_SIZE_FRACTION))
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    sarimax = fit_sarimax(y_train, order=(1,1,1))
    sarimax_pred = predict_sarimax(sarimax, steps=len(y_test))
    sarimax_mets = metrics(y_test, sarimax_pred)

    # XGBoost / LightGBM with rolling features
    ts = city_df[[DATE_COL, temp_col]].copy()
    ts = add_rolling_lags(ts, temp_col, ROLL_WINDOWS).reset_index(drop=True)
    split2 = int(len(ts) * (1 - TEST_SIZE_FRACTION))
    X = ts.drop(columns=[temp_col, DATE_COL])
    y2 = ts[temp_col].values
    X_train, X_test = X.iloc[:split2], X.iloc[split2:]
    y2_train, y2_test = y2[:split2], y2[split2:]

    xgb = fit_xgb(X_train, y2_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mets = metrics(y2_test, xgb_pred)

    lgb_mets = None
    lgb_pred = None
    lgb = fit_lgbm(X_train, y2_train)
    if lgb is not None:
        lgb_pred = lgb.predict(X_test)
        lgb_mets = metrics(y2_test, lgb_pred)

    # Simple ensemble on the aligned ML horizon (use models with same length as X_test)
    preds = {"XGB": xgb_pred}
    if lgb_pred is not None:
        preds["LGBM"] = lgb_pred
    # resize SARIMAX to the same length as X_test by naive trimming or interpolation
    if len(sarimax_pred) >= len(X_test):
        preds["SARIMAX"] = sarimax_pred[-len(X_test):]
    else:
        # pad if shorter
        pad = np.full(len(X_test)-len(sarimax_pred), sarimax_pred[-1])
        preds["SARIMAX"] = np.concatenate([sarimax_pred, pad])

    ens_pred = ensemble_preds(preds)
    ensemble_mets = metrics(y2_test, ens_pred)

    # Feature importance (Permutation on XGB)
    imp = get_permutation_importance(xgb, X_test, y2_test, n_repeats=5)
    fi_plot = None
    if not imp.empty:
        plt.figure(figsize=(8,5))
        imp.head(15).iloc[::-1].plot(kind="barh")
        plt.title("Permutation Importance (XGB) — top 15")
        fi_plot = FIG_DIR / "feature_importance.png"
        safe_savefig(fi_plot.as_posix())

    # 5) Spatial / Geographical Patterns (if country column exists)
    spatial_assets = []
    guess_country_cols = [c for c in df.columns if c in ("country","country_name","nation")]
    if guess_country_cols:
        country_col = guess_country_cols[0]
        country_avg_temp = average_by_country(df, country_col, temp_col)
        choro_out = FIG_DIR / "choropleth_temp.png"
        ok = choropleth_if_available(country_avg_temp, country_col, temp_col, choro_out.as_posix())
        spatial_assets.append(choro_out.as_posix() if ok else choro_out.with_suffix(".csv").as_posix())

    # 6) Save comparison charts
    # SARIMAX vs actual (time-aligned to test set of y)
    plt.figure(figsize=(10,4))
    plt.plot(range(len(y_test)), y_test, label="Actual")
    plt.plot(range(len(y_test)), sarimax_pred, label="SARIMAX")
    plt.title(f"{city} — SARIMAX Forecast vs Actual")
    plt.legend()
    safe_savefig((FIG_DIR / "sarimax_vs_actual.png").as_posix())

    # XGB vs actual
    plt.figure(figsize=(10,4))
    plt.plot(range(len(y2_test)), y2_test, label="Actual")
    plt.plot(range(len(y2_test)), xgb_pred, label="XGB")
    plt.title(f"{city} — XGB Forecast vs Actual")
    plt.legend()
    safe_savefig((FIG_DIR / "xgb_vs_actual.png").as_posix())

    # Ensemble vs actual
    plt.figure(figsize=(10,4))
    plt.plot(range(len(y2_test)), y2_test, label="Actual")
    plt.plot(range(len(y2_test)), ens_pred, label="Ensemble")
    plt.title(f"{city} — Ensemble Forecast vs Actual")
    plt.legend()
    safe_savefig((FIG_DIR / "ensemble_vs_actual.png").as_posix())

    # 7) Write report
    overview = {
        "Rows": len(df),
        "Cities detected": df[CITY_COL].nunique() if CITY_COL in df.columns else "N/A",
        "Example city": city,
        "Targets used": temp_col + (f", {PRECIP_COL}" if PRECIP_COL in df.columns else "")
    }

    model_results = {
        "SARIMAX": sarimax_mets,
        "XGBoost": xgb_mets
    }
    if lgb_mets:
        model_results["LightGBM"] = lgb_mets

    advanced_notes = {
        "anomalies_path": anomalies_plot.as_posix() if anomalies_plot else None,
        "feature_importance_path": fi_plot.as_posix() if fi_plot else None
    }

    write_report(
        overview=overview,
        eda_figs=[p.as_posix() for p in eda_figs],
        model_results=model_results,
        ensemble_metrics=ensemble_mets,
        advanced_notes=advanced_notes,
        spatial_assets=spatial_assets
    )

    print(f"\nDone. See report at: {REPORT_MD}\nFigures in: {FIG_DIR}\n")

if __name__== "__main__":
    main()
