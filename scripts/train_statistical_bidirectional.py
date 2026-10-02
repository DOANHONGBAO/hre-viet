"""Fit directional IBM1 + target bigram LMs on the fixed train split only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hre_translate.evaluation import evaluate_model  # noqa: E402
from hre_translate.models.statistical import StatisticalTranslator  # noqa: E402
from hre_translate.utils.config import load_config, resolve  # noqa: E402


def run(config_path: str = "configs/statistical.yaml") -> dict[str, dict]:
    config, root = load_config(config_path)
    train_path = resolve(root, config["data"]["train"])
    test_path = resolve(root, config["data"]["test"])
    if train_path.resolve() == test_path.resolve():
        raise ValueError("Train and test paths must differ")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path).reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Fixed train/test splits must be non-empty")
    dataset_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    reverse_metrics = {}
    for direction, settings in config["directions"].items():
        source_lang, target_lang = direction.split("_to_")
        options = {
            key: settings[key]
            for key in ("iterations", "lm_weight", "beam_size", "candidates_per_token")
        }
        model = StatisticalTranslator.from_frame(
            train, source_lang=source_lang, target_lang=target_lang, **options
        )
        artifact = resolve(root, settings["artifact"])
        model.save(artifact)
        (artifact / "provenance.json").write_text(
            json.dumps(
                {"train_split": config["data"]["train"], "train_sha256": dataset_hash,
                 "train_examples": len(train), "source_lang": source_lang,
                 "target_lang": target_lang, "settings": options},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8",
        )
        if direction != "vi_to_hre":
            continue
        metrics, predictions = evaluate_model(
            model, test, source_lang=source_lang, target_lang=target_lang
        )
        predictions_path = resolve(root, config["evaluation"]["reverse_predictions"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        result_path = resolve(root, config["evaluation"]["reverse_results"])
        reverse_metrics = {"model": "statistical", "direction": direction,
                           "train_examples": len(train), "test_examples": len(test),
                           "train_sha256": dataset_hash, "test_sha256": test_hash,
                           **metrics}
        pd.DataFrame([reverse_metrics]).to_csv(result_path, index=False, encoding="utf-8-sig")
        sample = predictions.sample(
            n=min(len(predictions), int(config["evaluation"]["sample_size"])),
            random_state=int(config["evaluation"]["sample_seed"]),
        )
        sample[["source", "reference", "hypothesis"]].rename(
            columns={"source": "source_vi", "reference": "reference_hre",
                     "hypothesis": "prediction_hre"}
        ).to_csv(resolve(root, config["evaluation"]["reverse_examples"]),
                 index=False, encoding="utf-8-sig")
    return reverse_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/statistical.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
