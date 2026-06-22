"""Data pipeline utilities for Predictive Analytics Using Historical Data."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from .predictive_models import DatasetConfig, generate_synthetic_historical_data, handle_missing_and_outliers, feature_engineering
from .data_cleaning_report import build_forecast_cleaning_report, write_forecast_cleaning_report





def make_dataset_and_features(
    cfg: DatasetConfig,
    max_lag: int = 14,
    *,
    artifacts_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate raw dataset and engineered dataset.

    If artifacts_dir is provided, write a data cleaning report JSON describing:
    - missing value imputation (median/mode)
    - outlier clipping bounds (1st-99th quantiles)
    """

    raw = generate_synthetic_historical_data(cfg)
    cleaned = handle_missing_and_outliers(raw)

    # Optional cleaning report
    if artifacts_dir:
        numeric_cols = [
            "numeric_feature_1",
            "numeric_feature_2",
            "numeric_feature_3",
            "target",
        ]
        categorical_cols = ["categorical_feature"]
        clip_cols = [
            "numeric_feature_1",
            "numeric_feature_2",
            "numeric_feature_3",
            "target",
        ]
        report = build_forecast_cleaning_report(
            raw,
            cleaned,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            clip_cols=clip_cols,
            clip_quantiles=(0.01, 0.99),
        )
        write_forecast_cleaning_report(
            report,
            os.path.join(artifacts_dir, "data_quality", "cleaning_report.json"),
        )

    engineered = feature_engineering(cleaned, max_lag=max_lag)
    return raw, engineered



def train_test_split_time_ordered(df_feat: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)
    n = len(df_feat)
    split = int(n * (1 - test_ratio))
    return df_feat.iloc[:split].copy(), df_feat.iloc[split:].copy()

