"""
Train the tool wear state classifier and RUL regressor, then save artifacts.

Usage
-----
    python scripts/train.py
    python scripts/train.py --data data/tool_wear_fleet.csv --output models/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import ToolWearGenerator
from predictor import ToolWearPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and save CNC tool wear models.")
    p.add_argument("--data",    type=str, default=None,       help="Path to existing CSV (generates fresh data if omitted)")
    p.add_argument("--n-tools", type=int, default=20,         help="Tools to generate if --data not provided")
    p.add_argument("--output",  type=str, default="models/",  help="Directory to save model artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data:
        logger.info("Loading data from %s", args.data)
        df = pd.read_csv(args.data)
    else:
        logger.info("Generating synthetic data for %d tools...", args.n_tools)
        df = ToolWearGenerator(n_tools=args.n_tools).generate()

    predictor = ToolWearPredictor()
    results = predictor.fit(df)

    print("\n=== Wear State Classification Report ===")
    print(results.classification_report)
    print(f"RUL Regressor — MAE: {results.rul_mae:.1f} cuts")
    print("\nTop feature importances:")
    print(results.feature_importances.tail(6).to_string())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, out_dir / "tool_wear_predictor.pkl")
    results.feature_importances.to_csv(out_dir / "feature_importances.csv")
    logger.info("Artifacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
