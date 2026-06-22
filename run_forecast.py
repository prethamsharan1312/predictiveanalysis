"""End-to-end runner for Predictive Analytics Using Historical Data.

Generates a synthetic historical dataset, performs preprocessing + feature
engineering, trains:
  1) Linear Regression baseline (supervised with lag/rolling features)
  2) Statsmodels SARIMAX (univariate forecasting)

Then evaluates with MAE/MSE/RMSE/R2 and saves artifacts + plots.

Usage:
  python3 src/run_forecast.py

Artifacts written to:
  artifacts/model_comparison.csv
  artifacts/metrics_summary.json
  artifacts/<plots>.png
"""

from __future__ import annotations

import os
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

import sys

# Allow running as: `python3 src/run_forecast.py`
# by adding the project root to sys.path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_pipeline import make_dataset_and_features

from src.predictive_models import (
    DatasetConfig,
    train_evaluate_linear_regression,
    train_evaluate_sarimax,
    save_artifacts,
)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_actual_vs_pred(ax, df_pred, title: str) -> None:
    ax.plot(df_pred["timestamp"], df_pred["actual"], label="Actual", linewidth=2)
    ax.plot(df_pred["timestamp"], df_pred["predicted"], label="Predicted", linewidth=2, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()


def plot_residuals(ax, df_pred, title: str) -> None:
    residuals = df_pred["actual"] - df_pred["predicted"]
    ax.scatter(df_pred["timestamp"], residuals, s=12, alpha=0.7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (Actual - Predicted)")


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_root, "artifacts")
    ensure_dir(artifacts_dir)

    # ---- Config ----
    cfg = DatasetConfig(n_records=1500, freq="D", start_date="2018-01-01", seed=42)
    max_lag = 14
    horizon = 7

    # ---- Prepare data ----
    raw_df, feat_df = make_dataset_and_features(cfg=cfg, max_lag=max_lag, artifacts_dir=artifacts_dir)



    # ---- Train + evaluate: Linear Regression ----
    lin_metrics = train_evaluate_linear_regression(feat_df, horizon=horizon)
    lin_pred_df = lin_metrics["predictions"]

    # ---- Train + evaluate: SARIMAX ----
    sarimax_metrics = train_evaluate_sarimax(feat_df, order=(1, 1, 1))
    sarimax_pred_df = sarimax_metrics["predictions"]

    # ---- Save artifacts/metrics ----
    comparisons = np.nan  # for type checkers
    comparisons = lin_pred_df[["timestamp", "actual", "predicted"]].rename(
        columns={"predicted": "predicted_linear"}
    )

    comparisons = comparisons.merge(
        sarimax_pred_df[["timestamp", "predicted"]].rename(columns={"predicted": "predicted_sarimax"}),
        on="timestamp",
        how="left",
    )

    # slim metrics for json
    lin_out = {k: v for k, v in lin_metrics.items() if k != "predictions"}
    sarimax_out = {k: v for k, v in sarimax_metrics.items() if k != "predictions"}

    # save summary json + csv using existing saver format
    model_metrics = {
        "dataset_config": asdict(cfg),
        "max_lag": max_lag,
        "horizon": horizon,
        "models": {
            "LinearRegression": lin_out,
            "Statsmodels_SARIMAX": sarimax_out,
        },
    }

    # Reuse existing save_artifacts signature by calling it for each model and writing
    # our own combined summary as well.
    save_artifacts(
        artifacts_dir=artifacts_dir,
        metrics={"LinearRegression": lin_out, **lin_out},
        comparisons=comparisons,
    )

    # Overwrite metrics_summary.json with a better structured summary
    import json

    with open(os.path.join(artifacts_dir, "metrics_summary.json"), "w") as f:
        json.dump(model_metrics, f, indent=2, default=str)

    comparisons.to_csv(os.path.join(artifacts_dir, "model_comparison.csv"), index=False)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_actual_vs_pred(axes[0, 0], lin_pred_df, "Linear Regression: Actual vs Predicted")
    plot_actual_vs_pred(axes[0, 1], sarimax_pred_df, "SARIMAX: Actual vs Predicted")

    plot_residuals(axes[1, 0], lin_pred_df, "Linear Regression Residuals")
    plot_residuals(axes[1, 1], sarimax_pred_df, "SARIMAX Residuals")

    plt.tight_layout()
    fig.savefig(os.path.join(artifacts_dir, "actual_vs_pred_and_residuals.png"), dpi=150)

    # ---- Console output ----
    print("=== Model metrics ===")
    print("LinearRegression:", {k: v for k, v in lin_out.items() if k != "predictions"})
    print("SARIMAX:", {k: v for k, v in sarimax_out.items() if k != "predictions"})
    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()

