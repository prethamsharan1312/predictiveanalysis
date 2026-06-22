from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastCleaningReport:
    dataset_rows_before: int
    dataset_rows_after: int

    numeric_imputed_median: Dict[str, Any]
    categorical_imputed_mode: Dict[str, Any]

    # outlier clip bounds used by pipeline
    outlier_clipping: Dict[str, Any]


def _safe_json(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def build_forecast_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    numeric_cols: List[str],
    categorical_cols: List[str],
    clip_cols: List[str],
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> ForecastCleaningReport:
    numeric_imputed_median: Dict[str, Any] = {}
    for c in numeric_cols:
        if c not in df_before.columns:
            continue
        missing_ct = int(df_before[c].isna().sum())
        if missing_ct == 0:
            continue
        median_val = float(pd.to_numeric(df_before[c], errors="coerce").median())
        numeric_imputed_median[c] = {
            "missing_filled_count": missing_ct,
            "strategy": "median",
            "median_value": median_val,
        }

    categorical_imputed_mode: Dict[str, Any] = {}
    for c in categorical_cols:
        if c not in df_before.columns:
            continue
        missing_ct = int(df_before[c].isna().sum())
        if missing_ct == 0:
            continue
        mode = df_before[c].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "Unknown"
        categorical_imputed_mode[c] = {
            "missing_filled_count": missing_ct,
            "strategy": "mode",
            "mode_value": _safe_json(fill),
        }

    lo_q, hi_q = clip_quantiles
    outlier_clipping: Dict[str, Any] = {}
    for c in clip_cols:
        if c not in df_before.columns:
            continue
        lo = float(pd.to_numeric(df_before[c], errors="coerce").quantile(lo_q))
        hi = float(pd.to_numeric(df_before[c], errors="coerce").quantile(hi_q))
        outlier_clipping[c] = {
            "strategy": "clip",
            "quantiles": [lo_q, hi_q],
            "lower_bound": lo,
            "upper_bound": hi,
        }

    return ForecastCleaningReport(
        dataset_rows_before=len(df_before),
        dataset_rows_after=len(df_after),
        numeric_imputed_median=numeric_imputed_median,
        categorical_imputed_mode=categorical_imputed_mode,
        outlier_clipping=outlier_clipping,
    )


def write_forecast_cleaning_report(report: ForecastCleaningReport, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report.__dict__, f, indent=2, default=_safe_json)

