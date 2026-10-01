from __future__ import annotations

import argparse
import json
from pathlib import Path

from hre_translate.neural.config import load_nllb_config
from hre_translate.neural.evaluate import evaluate_adapter
from hre_translate.neural.runtime import detect_runtime
from hre_translate.neural.training import train_nllb_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-600M with LoRA.")
    parser.add_argument("--config", type=Path, default=Path("configs/nllb.yaml"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    config, root = load_nllb_config(args.config)
    summary = train_nllb_lora(config, root)
    if not args.no_mlflow:
        from hre_translate.tracking import log_experiment

        result = evaluate_adapter(config, root, root / config.training.output_dir / "final_adapter")
        row = result.iloc[0]
        log_experiment(
            root=root,
            model_type="nllb_lora_training",
            metrics={
                key: row[key]
                for key in ("bleu", "chrf_pp", "latency_ms_mean", "latency_ms_p95")
            }
            | {"train_loss": summary["training_metrics"].get("train_loss", float("nan"))},
            parameters=config.model_dump(),
            datasets={
                "train": root / config.data.train,
                "validation": root / config.data.validation,
                "test": root / config.data.test,
            },
            config_path=root / args.config,
            artifacts=[
                root / config.training.output_dir / "training_summary.json",
                root / config.training.output_dir / "final_adapter" / "adapter_config.json",
                root / config.training.output_dir / "final_adapter" / "adapter_model.safetensors",
                root / config.evaluation.results,
                root / config.evaluation.predictions,
            ],
            latency_context="live_nllb_" + detect_runtime().device,
            extra_tags={"adapter_path": summary["adapter_path"]},
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
