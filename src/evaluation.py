"""
Step 4: Evaluation & Reporting for PM2.5 Time Series Project

This module:
- Loads fitted SARIMA model and makes predictions on test set
- Calculates RMSE, MAE, MAPE
- Plots Actual vs Predicted
- Residual diagnostics (Ljung-Box, histogram, Q-Q plot)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox

from sarima_model import (
    FACTOR_COLS,
    aggregate_to_daily,
    get_project_root,
    load_fa_data,
    train_test_split,
)


def load_sarima_model() -> dict:
    """Load saved SARIMA model bundle from joblib."""
    path = get_project_root() / "data" / "processed" / "sarima_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE."""
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
    """Plot Actual vs Predicted line chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_actual.index, y_actual.values, label="Actual", color="#2E86AB", linewidth=1)
    ax.plot(y_actual.index, y_pred, label="Predicted", color="#E94F37", linewidth=1, alpha=0.9)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("PM2.5 (μg/m³)", fontsize=11)
    ax.set_title("Actual vs Predicted PM2.5 (Test Set)", fontsize=14)
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
    """
    Create residual diagnostics: Ljung-Box test result, Residual Histogram, Q-Q plot.
    """
    residuals = np.asarray(residuals).flatten()
    residuals = residuals[~np.isnan(residuals)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Residual histogram
    axes[0].hist(residuals, bins=40, color="#2E86AB", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="#E94F37", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Histogram")
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    qqplot(residuals, line="45", ax=axes[1])
    axes[1].set_title("Q-Q Plot (Normality)")
    axes[1].grid(True, alpha=0.3)

    # Ljung-Box: plot p-values for lags 1 to ljungbox_lags
    lb_result = acorr_ljungbox(residuals, lags=range(1, ljungbox_lags + 1), return_df=True)
    lags = lb_result.index if hasattr(lb_result.index, "__len__") else range(1, ljungbox_lags + 1)
    axes[2].bar(lags, lb_result["lb_pvalue"], color="#44AF69", alpha=0.8)
    axes[2].axhline(0.05, color="#E94F37", linestyle="--", label="α=0.05")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Ljung-Box p-value")
    axes[2].set_title("Ljung-Box Test (Residual Autocorrelation)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Residual Diagnostics", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def run_evaluation_pipeline(
    test_ratio: float = 0.2,
    dpi: int = 300,
) -> dict:
    """
    Run full Step 4 pipeline:
    1. Load model, fa_data, aggregate to daily, train/test split
    2. Predict on test set
    3. Compute RMSE, MAE, MAPE -> save to evaluation_metrics.csv
    4. Plot Actual vs Predicted
    5. Residual diagnostics -> residual_diagnostics.png

    Returns metrics dict.
    """
    root = get_project_root()
    eval_fig = root / "reports" / "figures" / "04_eval"
    tables_dir = root / "reports" / "tables"

    print("Loading SARIMA model...")
    bundle = load_sarima_model()
    model = bundle["model"]

    print("Loading fa_data and preparing test set...")
    df = load_fa_data()
    daily = aggregate_to_daily(df)
    y_train, y_test, exog_train, exog_test = train_test_split(daily, test_ratio)

    # Predict on test set
    print("Predicting on test set...")
    exog_test_arr = exog_test.values if exog_test is not None else None
    y_pred = model.predict(n_periods=len(y_test), X=exog_test_arr)

    # Metrics
    metrics = compute_metrics(y_test.values, y_pred)
    print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")

    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "evaluation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    # Actual vs Predicted
    print("Plotting Actual vs Predicted...")
    plot_actual_vs_predicted(y_test, y_pred, eval_fig / "actual_vs_predicted.png", dpi=dpi)
    print(f"Saved to {eval_fig / 'actual_vs_predicted.png'}")

    # Save demo data for Streamlit (predictions + CI + residuals)
    try:
        pred, conf = model.predict(
            n_periods=len(y_test), X=exog_test.values if exog_test is not None else None,
            return_conf_int=True, alpha=0.05,
        )
        res = model.resid()
        demo_dir = root / "data" / "processed"
        demo_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "date": y_test.index,
            "actual": y_test.values,
            "predicted": pred,
            "ci_low": conf[:, 0],
            "ci_high": conf[:, 1],
        }).to_csv(demo_dir / "demo_predictions.csv", index=False)
        res_flat = np.asarray(res).flatten()[:5000]
        pd.DataFrame({"residual": res_flat}).to_csv(demo_dir / "demo_residuals.csv", index=False)
        import json
        with open(demo_dir / "demo_params.json", "w") as f:
            json.dump({"order": list(model.order), "seasonal_order": list(model.seasonal_order)}, f)
        print(f"Saved demo data to {demo_dir}")
    except Exception as e:
        print(f"Could not save demo data: {e}")

    # Residual diagnostics (use training residuals from fitted model)
    print("Running residual diagnostics...")
    residuals = model.resid()
    plot_residual_diagnostics(
        residuals,
        eval_fig / "residual_diagnostics.png",
        dpi=dpi,
    )
    print(f"Saved to {eval_fig / 'residual_diagnostics.png'}")

    return metrics


if __name__ == "__main__":
    run_evaluation_pipeline()
