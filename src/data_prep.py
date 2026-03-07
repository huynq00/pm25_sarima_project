"""
Step 1: Data Preprocessing for PM2.5 Time Series Project

This module:
- Loads raw CSV from data/raw/
- Creates datetime index
- Handles missing values via temporal interpolation
- Generates EDA plots (PM2.5 time series, correlation heatmap)
- Saves cleaned data to data/interim/cleaned_data.csv
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_project_root() -> Path:
    """Get project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_raw_data(csv_filename: str = None) -> pd.DataFrame:
    """
    Load raw CSV from data/raw/. If no filename given, uses first .csv found.

    Expected columns: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, Wd, WSPM, year, month, day, hour
    """
    raw_dir = get_project_root() / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    if csv_filename is None:
        csv_files = list(raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")
        csv_path = csv_files[0]
    else:
        csv_path = raw_dir / csv_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create datetime index from year, month, day, hour columns.
    Falls back to parsing if those columns don't exist.
    """
    df = df.copy()

    # Standard column names
    time_cols = ["year", "month", "day", "hour"]
    if all(c in df.columns for c in time_cols):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]].astype(int)
        )
        df = df.drop(columns=["year", "month", "day", "hour"], errors="ignore")
        df = df.set_index("datetime").sort_index()
        return df

    # Alternative: try common datetime column names
    for col in ["datetime", "date", "Date", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col).sort_index()
            return df

    raise ValueError(
        "Could not create datetime index. Need columns: year, month, day, hour OR datetime/date"
    )


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using temporal interpolation (linear for numeric).
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method="time")

    # If any remain (e.g., at boundaries), forward/backward fill
    df = df.ffill().bfill()

    return df


def save_eda_plots(df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> None:
    """
    Generate and save EDA plots:
    1. PM2.5 time series
    2. Correlation heatmap

    Saves to reports/figures/01_eda/ with specified dpi.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. PM2.5 Time Series ---
    if "PM2.5" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["PM2.5"], color="#2E86AB", linewidth=0.8, alpha=0.9)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("PM2.5 (μg/m³)", fontsize=11)
        ax.set_title("PM2.5 Time Series", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "pm25_time_series.png", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        # Try alternative naming
        pm_col = [c for c in df.columns if "pm" in c.lower() and "2" in c]
        if pm_col:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df[pm_col[0]], color="#2E86AB", linewidth=0.8, alpha=0.9)
            ax.set_xlabel("Date")
            ax.set_ylabel("PM2.5 (μg/m³)")
            ax.set_title("PM2.5 Time Series")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "pm25_time_series.png", dpi=dpi, bbox_inches="tight")
            plt.close()

    # --- 2. Correlation Heatmap ---
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def run_data_prep(csv_filename: str = None) -> pd.DataFrame:
    """
    Run full Step 1 pipeline:
    1. Load raw data
    2. Create datetime index
    3. Handle missing values
    4. Save EDA plots to reports/figures/01_eda/
    5. Save cleaned data to data/interim/cleaned_data.csv

    Returns the cleaned DataFrame.
    """
    root = get_project_root()
    eda_dir = root / "reports" / "figures" / "01_eda"
    interim_dir = root / "data" / "interim"

    print("Loading raw data...")
    df = load_raw_data(csv_filename)

    print("Creating datetime index...")
    df = create_datetime_index(df)

    print("Handling missing values (temporal interpolation)...")
    df = handle_missing_values(df)

    print("Generating EDA plots (dpi=300)...")
    save_eda_plots(df, eda_dir, dpi=300)

    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "cleaned_data.csv"
    df.to_csv(out_path)
    print(f"Saved cleaned data to {out_path}")

    return df


if __name__ == "__main__":
    run_data_prep()
