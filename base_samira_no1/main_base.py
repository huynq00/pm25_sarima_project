"""
Experiment 1 entrypoint: Baseline SARIMA (PM2.5 only).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "base" / "src"))

from evaluation_base import run_evaluation_pipeline_base
from sarima_model_base import run_sarima_pipeline_base


def main() -> None:
    print("=" * 60)
    print("Experiment 1 - Baseline SARIMA (PM2.5 only)")
    print("=" * 60)

    print("\n>>> Step 3 (Base): SARIMA only PM2.5")
    run_sarima_pipeline_base()

    print("\n>>> Step 4 (Base): Evaluation")
    run_evaluation_pipeline_base()

    print("\n" + "=" * 60)
    print("Baseline experiment completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
