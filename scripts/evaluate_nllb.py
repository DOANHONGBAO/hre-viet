from __future__ import annotations

import argparse
from pathlib import Path

from hre_translate.neural.config import load_nllb_config
from hre_translate.neural.evaluate import evaluate_adapter
from hre_translate.neural.runtime import detect_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained NLLB LoRA adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/nllb.yaml"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    config, root = load_nllb_config(args.config)
    adapter = args.adapter or root / config.training.output_dir / "final_adapter"
    result = evaluate_adapter(config, root, adapter)
    if not args.no_mlflow:
        from hre_translate.tracking import log_experiment

        row = result.iloc[0]
        log_experiment(
            root=root,
            model_type="nllb_lora",
            metrics={
                key: row[key]
                for key in ("bleu", "chrf_pp", "latency_ms_mean", "latency_ms_p95")
            },
            parameters=config.model_dump(),
            datasets={
                "train": root / config.data.train,
                "validation": root / config.data.validation,
                "test": root / config.data.test,
            },
            config_path=root / args.config,
            artifacts=[
                root / config.evaluation.results,
                root / config.evaluation.predictions,
            ],
            latency_context="live_nllb_" + detect_runtime().device,
            extra_tags={"adapter_path": str(adapter)},
        )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
