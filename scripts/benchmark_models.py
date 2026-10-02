"""Reproducible fixed-test comparison; neural rows are checked historical replay."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hre_translate.evaluation import evaluate_model  # noqa: E402
from hre_translate.models.dictionary import DictionaryTranslator  # noqa: E402
from hre_translate.models.retrieval import TranslationMemory  # noqa: E402
from hre_translate.models.rule_based import RuleBasedTranslator  # noqa: E402
from hre_translate.models.statistical import IBMModel1, StatisticalTranslator  # noqa: E402
from hre_translate.neural.metrics import corpus_metrics  # noqa: E402
from hre_translate.utils.config import load_config, resolve  # noqa: E402


def _checked_replay(path: Path, test: pd.DataFrame) -> pd.DataFrame:
    saved = pd.read_csv(path).fillna("")
    if len(saved) != len(test):
        raise ValueError(f"{path} does not contain the full fixed test set")
    for column, expected in (("source", test.hre), ("reference", test.vi)):
        if saved[column].astype(str).tolist() != expected.astype(str).tolist():
            raise ValueError(f"{path} differs from fixed test {column}")
    if "dataset_type" in saved and (
        saved.dataset_type.astype(str).tolist() != test.dataset_type.astype(str).tolist()
    ):
        raise ValueError(f"{path} differs from fixed test dataset_type")
    return saved


def benchmark(root: Path, config: dict) -> pd.DataFrame:
    train = pd.read_parquet(resolve(root, config["data"]["train"]))
    train_hash = hashlib.sha256(resolve(root, config["data"]["train"]).read_bytes()).hexdigest()
    test_hash = hashlib.sha256(resolve(root, config["data"]["test"]).read_bytes()).hexdigest()
    test = pd.read_parquet(resolve(root, config["data"]["test"])).reset_index(drop=True)
    if test.empty or not {"hre", "vi", "dataset_type"} <= set(test):
        raise ValueError("Fixed test split is missing or invalid")
    references = test.vi.astype(str).tolist()
    results = []
    hypotheses: dict[str, list[str]] = {}
    predictions_dir = resolve(root, config["evaluation"]["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    lexical_train = train.loc[train.dataset_type.isin(["lexicon", "phrase"])]
    models = {
        "rule_based": (
            RuleBasedTranslator.from_frame(lexical_train),
            "rule_based",
        ),
        "dictionary": (
            DictionaryTranslator.from_frame(lexical_train),
            "rule_based",
        ),
        "ibm1": (IBMModel1.from_frame(train), "statistical"),
        "statistical": (
            StatisticalTranslator.from_frame(train, **config["statistical"]),
            "statistical",
        ),
        "translation_memory": (
            TranslationMemory().fit(train.loc[train.dataset_type.eq("sentence")]),
            "retrieval",
        ),
    }
    statistical = models["statistical"][0]
    statistical.save(resolve(root, config["data"]["statistical_artifact"]))
    for name, (model, category) in models.items():
        metrics, predictions = evaluate_model(model, test)
        predictions.to_csv(predictions_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        hypotheses[name] = predictions.hypothesis.astype(str).tolist()
        results.append(
            {
                "model": name,
                "direction": "hre_to_vi",
                "category": category,
                "BLEU": metrics["bleu"],
                "chrF++": metrics["chrf_pp"],
                "avg_latency_ms": metrics["latency_ms_mean"],
                "median_latency_ms": metrics["latency_ms_median"],
                "parameters": "",
                "notes": "Live local test inference; train-only fit",
                "test_examples": len(test),
                "oov_rate": metrics["oov_rate"],
            }
        )
    saved_paths = {
        "nllb_lora": resolve(root, config["data"]["nllb_predictions"]),
        **{
            name: resolve(root, config["data"]["hybrid_predictions_dir"]) / f"{name}.csv"
            for name in ("nllb_only", "nllb_dictionary", "nllb_tm", "full_hybrid")
        },
    }
    for name, path in saved_paths.items():
        category = "pretrained" if name in {"nllb_lora", "nllb_only"} else "hybrid"
        saved = _checked_replay(path, test)
        candidate = saved.hypothesis.astype(str).tolist()
        hypotheses[name] = candidate
        metrics = corpus_metrics(candidate, references)
        if name == "nllb_lora":
            latencies = pd.to_numeric(saved.latency_ms, errors="raise")
            average = round(float(np.mean(latencies)), 6)
            median = round(float(np.median(latencies)), 6)
            notes = "Historical Kaggle T4 inference; not same-hardware latency"
        else:
            average = ""
            median = ""
            notes = "Validated offline neural replay; end-to-end latency unavailable"
        results.append(
            {
                "model": name,
                "direction": "hre_to_vi",
                "category": category,
                "BLEU": metrics["bleu"],
                "chrF++": metrics["chrf_pp"],
                "avg_latency_ms": average,
                "median_latency_ms": median,
                "parameters": "",
                "notes": notes,
                "test_examples": len(test),
                "oov_rate": "",
            }
        )
    reverse_path = resolve(root, config["data"]["reverse_results"])
    if reverse_path.is_file():
        reverse = pd.read_csv(reverse_path).iloc[0]
        if (reverse["train_sha256"] != train_hash
                or reverse["test_sha256"] != test_hash
                or int(reverse["test_examples"]) != len(test)):
            raise ValueError("Reverse benchmark is stale; rerun train_statistical_bidirectional.py")
        results.append({
            "model": "statistical", "direction": "vi_to_hre", "category": "statistical",
            "BLEU": reverse["bleu"], "chrF++": reverse["chrf_pp"],
            "avg_latency_ms": reverse["latency_ms_mean"],
            "median_latency_ms": reverse["latency_ms_median"],
            "parameters": "", "notes": "Live local reverse test inference; train-only fit",
            "test_examples": len(test), "oov_rate": reverse["oov_rate"],
        })
    comparison = pd.DataFrame(results).sort_values(["direction", "chrF++"], ascending=[True, False])
    comparison_path = resolve(root, config["evaluation"]["comparison"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    sample = test.sample(
        n=min(len(test), int(config["evaluation"]["sample_size"])),
        random_state=int(config["evaluation"]["sample_seed"]),
    )
    outputs = pd.DataFrame(
        {"source": sample.hre.astype(str).tolist(), "reference": sample.vi.astype(str).tolist()}
    )
    for name, values in hypotheses.items():
        outputs[name] = [values[position] for position in sample.index]
    outputs.to_csv(
        resolve(root, config["evaluation"]["outputs"]), index=False, encoding="utf-8-sig"
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stabilization.yaml")
    args = parser.parse_args()
    config, root = load_config(args.config)
    print(benchmark(root, config).to_string(index=False))


if __name__ == "__main__":
    main()
