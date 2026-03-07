"""
PM2.5 Time Series Analysis: Full Pipeline

Runs the complete methodology from Step 1 to Step 4:
  Step 1: Data Preprocessing (data_prep.py)
  Step 2: Factor Analysis (factor_analysis.py)
  Step 3: SARIMA Modeling (sarima_model.py)
  Step 4: Evaluation & Reporting (evaluation.py)
"""

import sys
from pathlib import Path

# Ensure src is on path when running main.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_prep import run_data_prep
from evaluation import run_evaluation_pipeline
from factor_analysis import run_factor_analysis_pipeline
from sarima_model import run_sarima_pipeline


def main() -> None:
    print("=" * 60)
    print("PM2.5 Time Series Analysis - Full Pipeline")
    print("=" * 60)

    print("\n>>> Step 1: Data Preprocessing")
    run_data_prep()

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
