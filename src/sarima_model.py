"""
Step 3: SARIMA Modeling for PM2.5 Time Series Project

This module:
- Reads fa_data.csv
- Aggregates to daily (baseline)
- Time series decomposition (Trend, Seasonality, Residuals)
- ADF test for stationarity
- ACF/PACF plots
- auto_arima with exogenous factors
- Train/test split, fit, save model
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

FACTOR_COLS = ["Factor1", "Factor2", "Factor3"]


def get_project_root() -> Path:
    """Get project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_fa_data() -> pd.DataFrame:
    """Load fa_data from data/processed/fa_data.csv."""
    path = get_project_root() / "data" / "processed" / "fa_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"FA data not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to daily (mean for numeric columns).
    Required for baseline SARIMA (faster, daily seasonality m=7).
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    daily = df[numeric_cols].resample("D").mean()
    return daily.dropna(how="all")


def run_decomposition(
    series: pd.Series, output_path: Path, period: int = 7, dpi: int = 300
) -> None:
    """
    Decompose PM2.5 into Trend, Seasonal, Residual.
    period=7 for daily data (weekly seasonality).
    """
    decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color="#2E86AB")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("PM2.5 Time Series Decomposition")
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
    """Run Augmented Dickey-Fuller test. Returns dict with test stats."""
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
    """Save ADF test results to text and CSV."""
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
    text_path = output_path.with_suffix(".txt")
    with open(text_path, "w") as f:
        f.write("\n".join(lines))
    # Also save key metrics to CSV
    csv_path = output_path.with_name("adf_results.csv")
    pd.DataFrame([{"adf_statistic": results["adf_statistic"], "p_value": results["p_value"]}]).to_csv(csv_path, index=False)


def plot_acf_pacf(
    series: pd.Series, output_path: Path, lags: int = 40, dpi: int = 300
) -> None:
    """Plot ACF and PACF side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.suptitle("ACF & PACF for PM2.5 (Daily)", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test. Returns (y_train, y_test, exog_train, exog_test).
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    y_train = train["PM2.5"]
    y_test = test["PM2.5"]
    exog_cols = [c for c in FACTOR_COLS if c in df.columns]
    exog_train = train[exog_cols] if exog_cols else None
    exog_test = test[exog_cols] if exog_cols else None
    return y_train, y_test, exog_train, exog_test


def run_sarima_pipeline(
    test_ratio: float = 0.2,
    seasonal_period: int = 7,
    dpi: int = 300,
) -> object:
    """
    Run full Step 3 pipeline:
    1. Load fa_data, aggregate to daily
    2. Decomposition plot
    3. ADF test, save results
    4. ACF/PACF plot
    5. auto_arima with exog (factors)
    6. Train/test split, fit, save model

    Returns the fitted ARIMA model.
    """
    root = get_project_root()
    arima_fig = root / "reports" / "figures" / "03_arima"
    tables_dir = root / "reports" / "tables"
    models_dir = root / "data" / "processed"

    print("Loading fa_data...")
    df = load_fa_data()

    print("Aggregating to daily...")
    daily = aggregate_to_daily(df)

    if "PM2.5" not in daily.columns:
        raise ValueError("PM2.5 column not found in fa_data")
    pm25 = daily["PM2.5"]

    # Decomposition
    print("Running time series decomposition...")
    decomp_path = arima_fig / "decomposition.png"
    run_decomposition(pm25, decomp_path, period=seasonal_period, dpi=dpi)
    print(f"Saved decomposition to {decomp_path}")

    # ADF test
    print("Running ADF test...")
    adf_results = run_adf_test(pm25)
    adf_path = tables_dir / "adf_results"
    save_adf_results(adf_results, adf_path)
    print(f"ADF p-value: {adf_results['p_value']:.6f}")
    print(f"Saved ADF results to {tables_dir}")

    # ACF/PACF
    print("Plotting ACF & PACF...")
    acf_path = arima_fig / "acf_pacf.png"
    plot_acf_pacf(pm25, acf_path, dpi=dpi)
    print(f"Saved ACF/PACF to {acf_path}")

    # Train/test split
    print("Splitting train/test...")
    y_train, y_test, exog_train, exog_test = train_test_split(daily, test_ratio)
    print(f"Train: {len(y_train)} days, Test: {len(y_test)} days")

    # auto_arima
    print("Running auto_arima (may take a few minutes)...")
    exog_arr = exog_train.values if exog_train is not None else None
    model = auto_arima(
        y_train,
        X=exog_arr,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    print(f"Best model: {model.order} x {model.seasonal_order}")

    # Save model (include exog info for prediction)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "sarima_model.joblib"
    joblib.dump(
        {
            "model": model,
            "exog_cols": FACTOR_COLS,
            "seasonal_period": seasonal_period,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")

    return model


if __name__ == "__main__":
    run_sarima_pipeline()
