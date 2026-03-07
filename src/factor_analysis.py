"""
Step 2: Factor Analysis for PM2.5 Time Series Project

This module:
- Reads cleaned_data.csv
- Selects environmental/meteorological variables (excludes PM2.5)
- Standardizes data and performs Factor Analysis (principal, varimax)
- Generates Scree plot to determine number of factors
- Extracts latent factors and appends to dataset
- Saves fa_data.csv to data/processed/
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """Get project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


# Environmental/meteorological variables for FA (exclude PM2.5, No, wd, station)
FA_VARS = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned data from data/interim/cleaned_data.csv."""
    path = get_project_root() / "data" / "interim" / "cleaned_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def select_fa_variables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select only environmental/meteorological variables for FA (exclude PM2.5).
    Returns (fa_data, full_df) - fa_data has only FA vars, full_df is original.
    """
    available = [c for c in FA_VARS if c in df.columns]
    if len(available) < 3:
        raise ValueError(
            f"Need at least 3 FA variables. Found: {available}. "
            f"Expected any of: {FA_VARS}"
        )
    fa_data = df[available].copy()
    return fa_data, df


def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize (z-score) the data. Returns (standardized_df, fitted_scaler)."""
    scaler = StandardScaler()
    values = scaler.fit_transform(df)
    std_df = pd.DataFrame(values, index=df.index, columns=df.columns)
    return std_df, scaler


def get_eigenvalues_for_scree(data_std: pd.DataFrame) -> np.ndarray:
    """Compute eigenvalues from correlation matrix for scree plot."""
    corr = data_std.corr()
    eigenvalues, _ = np.linalg.eig(corr)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    return eigenvalues


def plot_scree(eigenvalues: np.ndarray, output_path: Path, dpi: int = 300) -> None:
    """Generate and save Scree plot (eigenvalues vs component number)."""
    n = len(eigenvalues)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n + 1), eigenvalues, color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.plot(range(1, n + 1), eigenvalues, "o-", color="#E94F37", linewidth=2, markersize=8)
    ax.axhline(y=1, color="gray", linestyle="--", label="Kaiser criterion (λ=1)")
    ax.set_xlabel("Factor Number", fontsize=11)
    ax.set_ylabel("Eigenvalue", fontsize=11)
    ax.set_title("Scree Plot (Factor Analysis)", fontsize=14)
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def select_n_factors(eigenvalues: np.ndarray, min_factors: int = 2, max_factors: int = 5) -> int:
    """
    Select number of factors using Kaiser criterion (eigenvalues > 1),
    bounded by min_factors and max_factors.
    """
    kaiser = np.sum(eigenvalues > 1)
    n = np.clip(kaiser, min_factors, min(max_factors, len(eigenvalues)))
    return int(n)


def run_factor_analysis(
    data_std: pd.DataFrame,
    n_factors: int,
    method: str = "principal",
    rotation: str = "varimax",
) -> Tuple[FactorAnalyzer, np.ndarray]:
    """
    Fit Factor Analysis and return fitted model + factor scores.
    """
    fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
    fa.fit(data_std)
    scores = fa.transform(data_std)
    return fa, scores


def save_factor_loadings(
    fa: FactorAnalyzer,
    var_names: list,
    output_dir: Path,
    dpi: int = 300,
) -> str:
    """
    Save factor loadings to CSV, heatmap, and generate interpretation text.
    Returns path to interpretation summary.
    """
    loadings = fa.loadings_
    factor_cols = [f"Factor{i+1}" for i in range(loadings.shape[1])]
    loadings_df = pd.DataFrame(loadings, index=var_names, columns=factor_cols)

    # Save loadings CSV
    loadings_path = output_dir / "factor_loadings.csv"
    loadings_df.round(4).to_csv(loadings_path)
    loadings_df = loadings_df.round(3)

    # Save heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        loadings_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Factor Loadings (Varimax Rotation)", fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    heatmap_path = output_dir.parent / "figures" / "02_fa" / "factor_loadings_heatmap.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    # Generate interpretation: top variables per factor
    thresh = 0.4  # minimum |loading| to consider
    lines = [
        "KẾT LUẬN: CÁC NHÓM NGUYÊN NHÂN CHÍNH ẢNH HƯỞNG ĐẾN PM2.5",
        "=" * 60,
        "",
        "Dựa trên ma trận Factor Loadings (sau phép quay Varimax), mỗi nhân tố",
        "đại diện cho một nhóm biến môi trường/khí tượng có tương quan chặt với nhau.",
        "Các biến có |loading| > 0.4 được coi là đóng góp đáng kể vào nhân tố đó.",
        "",
        "---",
    ]
    var_desc = {
        "PM10": "bụi mịn PM10",
        "SO2": "lưu huỳnh đioxit",
        "NO2": "nitơ đioxit",
        "CO": "cacbon monoxit",
        "O3": "ozon",
        "TEMP": "nhiệt độ",
        "PRES": "áp suất",
        "DEWP": "điểm sương",
        "RAIN": "lượng mưa",
        "WSPM": "tốc độ gió",
    }
    for i, fc in enumerate(factor_cols):
        col = loadings_df[fc]
        sorted_idx = col.abs().sort_values(ascending=False).index
        top_vars = [v for v in sorted_idx if abs(col[v]) >= thresh][:5]
        if not top_vars:
            top_vars = list(sorted_idx[:3])
        vars_str = ", ".join(f"{v} ({col[v]:.2f})" for v in top_vars)
        desc_str = ", ".join(var_desc.get(v, v) for v in top_vars)
        lines.append(f"\n{fc}:")
        lines.append(f"  Biến chính: {vars_str}")
        lines.append(f"  Diễn giải: Nhóm {desc_str}")
    lines.extend([
        "",
        "---",
        "KẾT LUẬN TỔNG HỢP:",
        "Các nhân tố trên phản ánh các nguồn ảnh hưởng khác nhau đến PM2.5:",
        "- Yếu tố ô nhiễm (PM10, SO2, NO2, CO) thường tương quan với hoạt động",
        "  giao thông, công nghiệp, đốt nhiên liệu.",
        "- Yếu tố khí tượng (TEMP, PRES, DEWP, RAIN, WSPM) ảnh hưởng đến",
        "  khuếch tán và tích tụ ô nhiễm trong không khí.",
        "- O3 thường phản ánh phản ứng quang hóa, liên quan điều kiện ánh sáng.",
        "",
        "Các nhân tố này được đưa vào mô hình SARIMA dưới dạng biến ngoại sinh (exog)",
        "để cải thiện độ chính xác dự báo PM2.5.",
    ])
    interp_path = output_dir / "factor_interpretation.txt"
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(interp_path)


def run_factor_analysis_pipeline(
    n_factors: int = None,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Run full Step 2 pipeline:
    1. Load cleaned_data.csv
    2. Select FA variables (exclude PM2.5)
    3. Standardize
    4. Save Scree plot
    5. Determine n_factors (if not given) via Kaiser criterion
    6. Fit FA (principal, varimax), extract factors
    7. Append factors to dataset
    8. Save to data/processed/fa_data.csv

    Returns the final dataframe with factor columns.
    """
    root = get_project_root()
    fa_fig_dir = root / "reports" / "figures" / "02_fa"
    processed_dir = root / "data" / "processed"

    print("Loading cleaned data...")
    df = load_cleaned_data()

    print("Selecting FA variables (exclude PM2.5)...")
    fa_data, full_df = select_fa_variables(df)

    print("Standardizing data...")
    fa_std, _ = standardize(fa_data)

    # Scree plot (use eigenvalues from correlation matrix)
    print("Computing eigenvalues for Scree plot...")
    eigenvalues = get_eigenvalues_for_scree(fa_std)

    scree_path = fa_fig_dir / "scree_plot.png"
    print(f"Saving Scree plot to {scree_path} (dpi={dpi})...")
    plot_scree(eigenvalues, scree_path, dpi=dpi)

    if n_factors is None:
        n_factors = select_n_factors(eigenvalues)
        print(f"Selected n_factors={n_factors} (Kaiser criterion)")

    print(f"Running Factor Analysis (method=principal, rotation=varimax, n_factors={n_factors})...")
    fa, scores = run_factor_analysis(fa_std, n_factors, method="principal", rotation="varimax")

    # Save factor loadings, heatmap, and interpretation
    tables_dir = root / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    interp_path = save_factor_loadings(fa, list(fa_data.columns), tables_dir, dpi=dpi)
    print(f"Saved factor_loadings.csv, heatmap, and interpretation to {tables_dir}")

    # Append factor scores to full dataframe
    factor_cols = [f"Factor{i+1}" for i in range(n_factors)]
    scores_df = pd.DataFrame(scores, index=full_df.index, columns=factor_cols)
    result = pd.concat([full_df, scores_df], axis=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "fa_data.csv"
    result.to_csv(out_path)
    print(f"Saved fa_data to {out_path}")

    return result


if __name__ == "__main__":
    run_factor_analysis_pipeline()
