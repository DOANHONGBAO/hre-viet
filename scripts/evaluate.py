from __future__ import annotations

import argparse
from pathlib import Path

from hre_translate.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 classical baselines.")
    parser.add_argument(
        "--model",
        choices=["dictionary", "translation_memory", "ibm1", "all"],
        required=True,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    models = ["dictionary", "translation_memory", "ibm1"] if args.model == "all" else [args.model]
    print(run_evaluation(models, args.config, track=not args.no_mlflow).to_string(index=False))


if __name__ == "__main__":
    main()
