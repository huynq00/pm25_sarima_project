"""
PM2.5 Time Series Analysis: Full Pipeline

Step 1 = EDA notebook (chạy trước, tạo cleaned_data.csv).
Step 2–4: Factor Analysis → SARIMA → Evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evaluation import run_evaluation_pipeline
from factor_analysis import run_factor_analysis_pipeline
from sarima_model import run_sarima_pipeline


def main() -> None:
    root = Path(__file__).resolve().parent
    cleaned_path = root / "data" / "interim" / "cleaned_data.csv"

    print("=" * 60)
    print("PM2.5 Time Series Analysis - Full Pipeline")
    print("=" * 60)

    print("\n>>> Step 1: Data (EDA + tiền xử lý)")
    if not cleaned_path.exists():
        print("ERROR: Chưa có cleaned_data.csv. Hãy chạy notebook notebooks/01_EDA.ipynb trước.")
        sys.exit(1)
    print("Dùng cleaned_data.csv từ EDA notebook. Bỏ qua Step 1.")

    print("\n>>> Step 2: Factor Analysis")
    run_factor_analysis_pipeline()

    print("\n>>> Step 3: SARIMA Modeling")
    run_sarima_pipeline()

    print("\n>>> Step 4: Evaluation & Reporting")
    run_evaluation_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
