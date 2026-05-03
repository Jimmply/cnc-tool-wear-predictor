"""
Generate and save synthetic tool wear fleet data.

Usage
-----
    python scripts/generate_data.py
    python scripts/generate_data.py --n-tools 50 --seed 123 --output data/fleet.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import ToolConfig, ToolWearGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic CNC tool wear data.")
    p.add_argument("--n-tools",  type=int,   default=20,  help="Number of tools to simulate")
    p.add_argument("--max-cuts", type=int,   default=500, help="Nominal tool life in cuts")
    p.add_argument("--seed",     type=int,   default=42,  help="Random seed")
    p.add_argument("--output",   type=str,   default="data/tool_wear_fleet.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ToolConfig(max_cuts=args.max_cuts)
    gen = ToolWearGenerator(n_tools=args.n_tools, tool_config=cfg, random_seed=args.seed)
    df = gen.generate()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info("Saved %d records to %s", len(df), out_path)
    logger.info("Wear state distribution:\n%s", df["wear_state"].value_counts().to_string())


if __name__ == "__main__":
    main()
