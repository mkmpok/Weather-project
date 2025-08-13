from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.config import REPORT_MD, PM_MISSION

def write_report(
    overview: Dict,
    eda_figs: List[str],
    model_results: Dict[str, Dict[str, float]],
    ensemble_metrics: Dict[str, float],
    advanced_notes: Dict,
    spatial_assets: List[str],
):
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    def fig_line(p):
        return f"![figure]({Path(p).as_posix()})"

    lines = []
    lines.append("# Weather Trend Forecasting — Report")
    lines.append("")
    lines.append("> *PM Accelerator Mission*")
    lines.append(f"> {PM_MISSION}")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for k,v in overview.items():
        lines.append(f"- *{k}:* {v}")
    lines.append("")
    lines.append("## 2. Data Cleaning & Preprocessing")
    lines.append("- Missing values: interpolation + median fallback")
    lines.append("- Outliers: clipped to 1st–99th percentile (numeric)")
    lines.append("- Normalization: z-score (numeric)")
    lines.append("- Time index: lastupdated")
    lines.append("")
    lines.append("## 3. Exploratory Data Analysis (EDA)")
    for p in eda_figs:
        lines.append(fig_line(p))
    lines.append("")
    lines.append("## 4. Forecasting Models & Metrics")
    for name, mets in model_results.items():
        lines.append(f"### {name}")
        for mk, mv in mets.items():
            lines.append(f"- {mk}: {mv:.4f}")
        lines.append("")
    lines.append("### Ensemble (simple average)")
    for mk, mv in ensemble_metrics.items():
        lines.append(f"- {mk}: {mv:.4f}")
    lines.append("")
    lines.append("## 5. Advanced Analyses")
    if advanced_notes.get("anomalies_path"):
        lines.append("### Anomaly Detection (IsolationForest)")
        lines.append(fig_line(advanced_notes["anomalies_path"]))
    if advanced_notes.get("feature_importance_path"):
        lines.append("### Feature Importance (Permutation)")
        lines.append(fig_line(advanced_notes["feature_importance_path"]))
    if spatial_assets:
        lines.append("### Spatial / Geographical Patterns")
        for p in spatial_assets:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("## 6. Conclusions & Next Steps")
    lines.append("- Summarize insights and model performance across cities.")
    lines.append("- Potential improvements: hyperparameter tuning, weather exogenous variables, hierarchical models.")
    lines.append("")
    Path(REPORT_MD).write_text("\n".join(lines), encoding="utf-8")