"""
Experiment 1 - Baseline SARIMA (PM2.5 only, no exogenous factors).

This module:
- Reads fa_data.csv
- Aggregates to daily
- Decomposition, ADF, ACF/PACF
- auto_arima WITHOUT exogenous variables
- Train/test split and save baseline model
"""

from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def get_project_root() -> Path:
    """Get project root (parent of base/)."""
    return Path(__file__).resolve().parents[2]


def load_fa_data() -> pd.DataFrame:
    """Load fa_data from original processed folder."""
    path = get_project_root() / "data" / "processed" / "fa_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"FA data not found: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily mean."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    daily = df[numeric_cols].resample("D").mean()
    return daily.dropna(how="all")


def run_decomposition(
    series: pd.Series, output_path: Path, period: int = 7, dpi: int = 300
) -> None:
    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color="#2E86AB")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("PM2.5 Time Series Decomposition (Baseline)")
    decomp.trend.plot(ax=axes[1], color="#E94F37")
    axes[1].set_ylabel("Trend")
    decomp.seasonal.plot(ax=axes[2], color="#44AF69")
    axes[2].set_ylabel("Seasonal")
    decomp.resid.plot(ax=axes[3], color="#8B8B8B")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def run_adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "usedlag": result[2],
        "nobs": result[3],
        "critical_values": result[4],
        "icbest": result[5],
    }


def save_adf_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Augmented Dickey-Fuller Test Results",
        "=" * 40,
        f"ADF Statistic: {results['adf_statistic']:.6f}",
        f"p-value:       {results['p_value']:.6f}",
        f"Used lag:      {results['usedlag']}",
        f"N observations: {results['nobs']}",
        "Critical values:",
    ]
    for k, v in results["critical_values"].items():
        lines.append(f"  {k}: {v:.3f}")
    lines.append("")
    lines.append("Conclusion: Series is STATIONARY" if results["p_value"] < 0.05 else "Conclusion: Series is NON-STATIONARY")

    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    pd.DataFrame([{"adf_statistic": results["adf_statistic"], "p_value": results["p_value"]}]).to_csv(
        output_path.with_name("adf_results.csv"), index=False
    )


def plot_acf_pacf(
    series: pd.Series, output_path: Path, lags: int = 40, dpi: int = 300
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.suptitle("ACF & PACF for PM2.5 Daily (Baseline)", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def train_test_split_pm25(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """Split PM2.5 only (no exogenous features)."""
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    y_train = train["PM2.5"]
    y_test = test["PM2.5"]
    return y_train, y_test


def run_sarima_pipeline_base(
    test_ratio: float = 0.2,
    seasonal_period: int = 7,
    dpi: int = 300,
) -> object:
    """
    Baseline SARIMA pipeline (PM2.5 only):
    - no exog in split
    - no X passed to auto_arima/predict
    """
    root = get_project_root()
    base_root = root / "base"
    arima_fig = base_root / "reports" / "figures" / "03_arima_base"
    tables_dir = base_root / "reports" / "tables"
    models_dir = base_root / "data" / "processed"

    print("Loading fa_data...")
    df = load_fa_data()
    print("Aggregating to daily...")
    daily = aggregate_to_daily(df)

    if "PM2.5" not in daily.columns:
        raise ValueError("PM2.5 column not found in fa_data")
    pm25 = daily["PM2.5"]

    print("Running time series decomposition...")
    run_decomposition(pm25, arima_fig / "decomposition.png", period=seasonal_period, dpi=dpi)

    print("Running ADF test...")
    adf_results = run_adf_test(pm25)
    save_adf_results(adf_results, tables_dir / "adf_results")
    print(f"ADF p-value: {adf_results['p_value']:.6f}")

    print("Plotting ACF & PACF...")
    plot_acf_pacf(pm25, arima_fig / "acf_pacf.png", dpi=dpi)

    print("Splitting train/test (PM2.5 only)...")
    y_train, y_test = train_test_split_pm25(daily, test_ratio)
    print(f"Train: {len(y_train)} days, Test: {len(y_test)} days")

    print("Running auto_arima baseline (no exog)...")
    model = auto_arima(
        y_train,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    print(f"Best model: {model.order} x {model.seasonal_order}")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "sarima_model_base.joblib"
    joblib.dump(
        {
            "model": model,
            "seasonal_period": seasonal_period,
            "experiment": "baseline_pm25_only",
        },
        model_path,
    )
    print(f"Saved baseline model to {model_path}")
    return model


if __name__ == "__main__":
    run_sarima_pipeline_base()
