"""Predictive Analytics Using Historical Data

This module provides:
- Synthetic dataset generation (>=1000 rows)
- Data cleaning (missing values + outlier clipping)
- Feature engineering (calendar, lags, rolling stats)
- Model training & evaluation for:
    1) Linear Regression (supervised baseline)
    2) Statsmodels time-series forecasting (SARIMAX)

The notebook/script is intended for portfolio submission, with metrics:
MAE, MSE, RMSE, R².
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class DatasetConfig:
    n_records: int = 1500
    freq: str = "D"  # daily
    start_date: str = "2018-01-01"
    seed: int = 42


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_synthetic_historical_data(cfg: DatasetConfig) -> pd.DataFrame:
    """Create a realistic time series with seasonality + trend + noise.

    Output columns:
    - timestamp (datetime)
    - target (float): value to forecast
    - numeric_feature_1 (float)
    - numeric_feature_2 (float)
    - numeric_feature_3 (float)
    - categorical_feature (str)

    We also introduce missing values and outliers to demonstrate cleaning.
    """

    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_records, freq=cfg.freq)

    # Calendar effects
    day_of_year = dates.dayofyear.values
    day_of_week = dates.dayofweek.values

    # Latent trend + seasonality
    trend = np.linspace(0.0, 25.0, cfg.n_records)
    seasonality = 10.0 * np.sin(2 * np.pi * day_of_year / 365.0)
    weekly = 3.0 * np.sin(2 * np.pi * day_of_week / 7.0)

    # Exogenous numeric drivers
    numeric_feature_1 = 50 + 0.08 * trend + rng.normal(0, 2.0, cfg.n_records)
    numeric_feature_2 = 30 + 0.05 * trend + 5 * np.sin(2 * np.pi * day_of_year / 180.0) + rng.normal(
        0, 1.8, cfg.n_records
    )
    numeric_feature_3 = 100 + rng.normal(0, 6.0, cfg.n_records)

    # Categorical driver (e.g., region/segment) repeating
    categories = np.array(["A", "B", "C", "D"])
    categorical_feature = categories[(dates.month.values // 3) % len(categories)]
    cat_effect = pd.Series(categorical_feature).map({"A": 1.5, "B": -2.0, "C": 3.5, "D": 0.0}).values

    # Target as combination
    noise = rng.normal(0, 2.5, cfg.n_records)
    target = 0.25 * numeric_feature_1 + 0.45 * numeric_feature_2 + 0.10 * numeric_feature_3
    target = target + trend * 0.9 + seasonality + weekly + cat_effect + noise

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "target": target,
            "numeric_feature_1": numeric_feature_1,
            "numeric_feature_2": numeric_feature_2,
            "numeric_feature_3": numeric_feature_3,
            "categorical_feature": categorical_feature,
        }
    )

    # Introduce missing values (~2%) in numeric columns
    for col in ["numeric_feature_1", "numeric_feature_2", "numeric_feature_3"]:
        mask = rng.random(cfg.n_records) < 0.02
        df.loc[mask, col] = np.nan

    # Introduce missing values in categorical (~1%)
    mask_cat = rng.random(cfg.n_records) < 0.01
    df.loc[mask_cat, "categorical_feature"] = np.nan

    # Introduce outliers in numeric_feature_2 (~1%)
    outlier_mask = rng.random(cfg.n_records) < 0.01
    df.loc[outlier_mask, "numeric_feature_2"] *= rng.choice([1.8, 2.2], size=outlier_mask.sum())

    return df


def handle_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values:
    - numeric columns: median
    - categorical: mode (or 'Unknown')

    Outliers:
    - clip numeric columns to [1st, 99th] percentiles
      (a robust approach to dampen extreme values)
    """

    df = df.copy()

    # Missing
    num_cols = ["numeric_feature_1", "numeric_feature_2", "numeric_feature_3", "target"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    if "categorical_feature" in df.columns:
        mode = df["categorical_feature"].mode(dropna=True)
        fill_val = mode.iloc[0] if len(mode) else "Unknown"
        df["categorical_feature"] = df["categorical_feature"].fillna(fill_val)

    # Outliers (clip)
    clip_cols = ["numeric_feature_1", "numeric_feature_2", "numeric_feature_3", "target"]
    for c in clip_cols:
        lo = df[c].quantile(0.01)
        hi = df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)

    return df


def feature_engineering(df: pd.DataFrame, max_lag: int = 14) -> pd.DataFrame:
    """Create supervised learning features.

    Added features:
    - calendar: day_of_week, month
    - lags: target_lag_1..target_lag_{max_lag}
    - rolling stats on target: rolling_mean_7, rolling_mean_14, rolling_std_7
    - encode categorical_feature as numeric via target-mean encoding (lightweight)

    Returns a dataframe ready for supervised modeling (rows with NaNs from lagging dropped).
    """

    df = df.copy().sort_values("timestamp")

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Lags
    for lag in range(1, max_lag + 1):
        df[f"target_lag_{lag}"] = df["target"].shift(lag)

    # Rolling stats
    df["rolling_mean_7"] = df["target"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["target"].shift(1).rolling(14).mean()
    df["rolling_std_7"] = df["target"].shift(1).rolling(7).std()

    # Categorical encoding: target mean per category (fit on past-ish, but simple for demo)
    category_target_mean = df.groupby("categorical_feature")["target"].mean()
    df["categorical_feature_encoded"] = df["categorical_feature"].map(category_target_mean)

    # Drop rows with missing due to lagging
    df = df.dropna().reset_index(drop=True)

    return df


def train_evaluate_linear_regression(df_feat: pd.DataFrame, horizon: int = 7) -> Dict:
    """Train LinearRegression to predict target at t+horizon using features at time t.

    Approach:
    - Create label y = target shifted -horizon
    - Use feature columns excluding timestamp and original target
    - Split by time: last 20% for test
    """

    df = df_feat.copy().sort_values("timestamp")

    # Label
    df["y"] = df["target"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    # Time split
    n = len(df)
    split = int(n * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    drop_cols = {"timestamp", "target", "categorical_feature", "y"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = compute_regression_metrics(y_test, pred)

    # Assemble actual vs predicted
    results = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].values,
            "actual": y_test.values,
            "predicted": pred,
        }
    )

    metrics_out = {
        "model": "LinearRegression",
        **metrics,
        "predictions": results,
    }
    return metrics_out


def train_evaluate_sarimax(df_feat: pd.DataFrame, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
    """Time-series forecasting with Statsmodels SARIMAX.

    We forecast a test window equal to last 20% of the engineered dataframe.
    SARIMAX uses only the univariate target series here for stability.
    """

    df = df_feat.copy().sort_values("timestamp")

    # Univariate series
    y = df["target"].astype(float)
    dates = df["timestamp"]

    n = len(df)
    split = int(n * 0.8)
    train_y = y.iloc[:split]
    test_y = y.iloc[split:]
    test_dates = dates.iloc[split:]

    # SARIMAX (no seasonality for generality; can be extended)
    model = sm.tsa.SARIMAX(train_y, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    pred = res.forecast(steps=len(test_y))

    metrics = compute_regression_metrics(test_y, pred)

    results = pd.DataFrame(
        {
            "timestamp": test_dates.values,
            "actual": test_y.values,
            "predicted": np.asarray(pred),
        }
    )

    return {
        "model": "Statsmodels_SARIMAX",
        "sarimax_order": order,
        **metrics,
        "predictions": results,
    }


def compute_regression_metrics(y_true, y_pred) -> Dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def save_artifacts(artifacts_dir: str, metrics: Dict, comparisons: pd.DataFrame) -> None:
    ensure_dir(artifacts_dir)

    # metrics without embedding huge predictions
    metrics_slim = {k: v for k, v in metrics.items() if k != "predictions"}

    with open(os.path.join(artifacts_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_slim, f, indent=2, default=str)

    comparisons.to_csv(os.path.join(artifacts_dir, "model_comparison.csv"), index=False)

