from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATASET, RANDOM_SEED, TARGET_COL
from src.run_classification import run as run_classification
from src.run_clustering import run as run_clustering
from src.run_dimensionality import run as run_dimensionality
from src.run_eda import run as run_eda


def main():
    parser = argparse.ArgumentParser(description="Run full ML pipeline")
    parser.add_argument("--data-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run_eda(args.data_path, target_col=args.target_col)
    run_dimensionality(args.data_path, target_col=args.target_col, random_state=args.random_state)
    run_clustering(args.data_path, target_col=args.target_col, random_state=args.random_state)
    run_classification(
        args.data_path,
        target_col=args.target_col,
        random_state=args.random_state,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
