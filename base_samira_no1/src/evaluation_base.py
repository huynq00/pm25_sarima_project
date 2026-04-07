"""
Experiment 1 - Baseline SARIMA evaluation (PM2.5 only).
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox

from sarima_model_base import (
    aggregate_to_daily,
    get_project_root,
    load_fa_data,
    train_test_split_pm25,
)


def load_sarima_model_base() -> dict:
    path = get_project_root() / "base" / "data" / "processed" / "sarima_model_base.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Baseline model not found: {path}")
    return joblib.load(path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def plot_actual_vs_predicted(
    y_actual: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_actual.index, y_actual.values, label="Actual", color="#2E86AB", linewidth=1)
    ax.plot(y_actual.index, y_pred, label="Predicted (Baseline)", color="#E94F37", linewidth=1, alpha=0.9)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("PM2.5 (μg/m3)", fontsize=11)
    ax.set_title("Actual vs Predicted PM2.5 (Baseline Test Set)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_residual_diagnostics(
    residuals: np.ndarray,
    output_path: Path,
    ljungbox_lags: int = 15,
    dpi: int = 300,
) -> None:
    residuals = np.asarray(residuals).flatten()
    residuals = residuals[~np.isnan(residuals)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(residuals, bins=40, color="#2E86AB", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="#E94F37", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Histogram")
    axes[0].grid(True, alpha=0.3)

    qqplot(residuals, line="45", ax=axes[1])
    axes[1].set_title("Q-Q Plot (Normality)")
    axes[1].grid(True, alpha=0.3)

    lb_result = acorr_ljungbox(residuals, lags=range(1, ljungbox_lags + 1), return_df=True)
    lags = lb_result.index if hasattr(lb_result.index, "__len__") else range(1, ljungbox_lags + 1)
    axes[2].bar(lags, lb_result["lb_pvalue"], color="#44AF69", alpha=0.8)
    axes[2].axhline(0.05, color="#E94F37", linestyle="--", label="alpha=0.05")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Ljung-Box p-value")
    axes[2].set_title("Ljung-Box Test (Residual Autocorrelation)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Residual Diagnostics (Baseline)", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def run_evaluation_pipeline_base(
    test_ratio: float = 0.2,
    dpi: int = 300,
) -> dict:
    root = get_project_root()
    base_root = root / "base"
    eval_fig = base_root / "reports" / "figures" / "04_eval_base"
    tables_dir = base_root / "reports" / "tables"
    data_dir = base_root / "data" / "processed"

    print("Loading baseline SARIMA model...")
    bundle = load_sarima_model_base()
    model = bundle["model"]

    print("Loading fa_data and preparing test set...")
    df = load_fa_data()
    daily = aggregate_to_daily(df)
    _, y_test = train_test_split_pm25(daily, test_ratio)

    print("Predicting on test set (no exog)...")
    y_pred = model.predict(n_periods=len(y_test))

    metrics = compute_metrics(y_test.values, y_pred)
    print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")

    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(tables_dir / "evaluation_metrics_base.csv", index=False)
    print(f"Saved metrics to {tables_dir / 'evaluation_metrics_base.csv'}")

    print("Plotting Actual vs Predicted...")
    plot_actual_vs_predicted(y_test, y_pred, eval_fig / "actual_vs_predicted_base.png", dpi=dpi)

    try:
        pred, conf = model.predict(n_periods=len(y_test), return_conf_int=True, alpha=0.05)
        residuals = model.resid()
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": y_test.index,
                "actual": y_test.values,
                "predicted": pred,
                "ci_low": conf[:, 0],
                "ci_high": conf[:, 1],
            }
        ).to_csv(data_dir / "demo_predictions_base.csv", index=False)
        pd.DataFrame({"residual": np.asarray(residuals).flatten()[:5000]}).to_csv(
            data_dir / "demo_residuals_base.csv", index=False
        )
        import json

        with open(data_dir / "demo_params_base.json", "w", encoding="utf-8") as f:
            json.dump({"order": list(model.order), "seasonal_order": list(model.seasonal_order)}, f)
    except Exception as e:
        print(f"Could not save baseline demo data: {e}")

    print("Running residual diagnostics...")
    plot_residual_diagnostics(model.resid(), eval_fig / "residual_diagnostics_base.png", dpi=dpi)

    return metrics


if __name__ == "__main__":
    run_evaluation_pipeline_base()
