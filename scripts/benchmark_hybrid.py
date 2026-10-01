"""Stage 4 fixed-test benchmark using saved Kaggle NLLB hypotheses.

Replay avoids claiming that CPU timings reproduce the Kaggle T4 run. The same
saved neural hypothesis is shared across ablations; no test reference is used
by routing or post-editing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hre_translate.hybrid import HybridTranslator  # noqa: E402
from hre_translate.models.dictionary import DictionaryTranslator  # noqa: E402
from hre_translate.models.retrieval import TranslationMemory  # noqa: E402
from hre_translate.neural.metrics import corpus_metrics  # noqa: E402
from hre_translate.utils.config import load_config, resolve  # noqa: E402


class SavedNLLB:
    """Sequential, source-checked replay of genuine Kaggle test predictions."""

    def __init__(self, predictions: pd.DataFrame) -> None:
        self.predictions = predictions.reset_index(drop=True)
        self.position = 0

    def translate(self, text: str) -> dict[str, Any]:
        if self.position >= len(self.predictions):
            raise ValueError("NLLB prediction replay exhausted")
        row = self.predictions.iloc[self.position]
        if str(row.source) != text:
            raise ValueError(f"NLLB replay source mismatch at row {self.position}")
        self.position += 1
        return {"translation": str(row.hypothesis)}

    def skip(self, text: str) -> None:
        """Advance when TM routes around NLLB; still verify source alignment."""
        self.translate(text)


def validate_replay(test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if len(test) != len(predictions):
        raise ValueError("NLLB predictions must cover every test row")
    for column, expected in (
        ("source", test.hre),
        ("reference", test.vi),
        ("dataset_type", test.dataset_type),
    ):
        if predictions[column].astype(str).tolist() != expected.astype(str).tolist():
            raise ValueError(f"NLLB predictions do not match fixed test {column}")
    if predictions.hypothesis.isna().any():
        raise ValueError("NLLB replay contains missing hypotheses")


def benchmark(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    train = pd.read_parquet(resolve(root, config["data"]["train"]))
    test = pd.read_parquet(resolve(root, config["data"]["test"]))
    saved = pd.read_csv(resolve(root, config["data"]["nllb_predictions"]))
    validate_replay(test, saved)
    lexicon = DictionaryTranslator.from_frame(
        train.loc[train.dataset_type.isin(config["lexicon_types"])]
    )
    options = config["retrieval"]
    memory = TranslationMemory(
        ngram_range=tuple(options["ngram_range"]),
        direct_match_threshold=float(options["high_threshold"]),
    ).fit(train.loc[train.dataset_type.isin(config["memory_types"])])
    output_dir = resolve(root, config["evaluation"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    references = test.vi.astype(str).tolist()
    tm_rows = []
    for row in test.itertuples(index=False):
        started = time.perf_counter()
        answer = memory.translate(str(row.hre), top_k=int(options["top_k"]))
        elapsed = (time.perf_counter() - started) * 1000
        tm_rows.append(
            {
                "dataset_type": row.dataset_type,
                "source": row.hre,
                "reference": row.vi,
                "hypothesis": answer["translation"],
                "model": "translation_memory",
                "similarity": answer["retrievals"][0]["similarity"],
                "replay_processing_ms": elapsed,
            }
        )
    tm_predictions = pd.DataFrame(tm_rows)
    tm_predictions.to_csv(output_dir / "translation_memory.csv", index=False, encoding="utf-8-sig")
    results.append(
        {
            "model": "translation_memory",
            "test_examples": len(test),
            **corpus_metrics(tm_predictions.hypothesis.astype(str).tolist(), references),
            "replay_processing_ms_mean": round(
                float(np.mean(tm_predictions.replay_processing_ms)), 6
            ),
            "replay_processing_ms_p95": round(
                float(np.percentile(tm_predictions.replay_processing_ms, 95)), 6
            ),
            "tm_routes": len(test),
            "medium_tm_fallbacks": 0,
            "dictionary_edits": 0,
            "latency_note": "Local live TM retrieval; no NLLB generation",
        }
    )
    for variant in ("nllb_only", "nllb_dictionary", "nllb_tm", "full_hybrid"):
        neural = SavedNLLB(saved)
        service = HybridTranslator(
            lexicon,
            memory,
            neural,
            medium_threshold=float(options["medium_threshold"]),
            high_threshold=float(options["high_threshold"]),
            top_k=int(options["top_k"]),
            variant=variant,
        )
        rows = []
        for index, row in enumerate(test.itertuples(index=False)):
            before = neural.position
            result = service.translate(str(row.hre))
            if neural.position == before:
                neural.skip(str(row.hre))
            rows.append(
                {
                    "dataset_type": row.dataset_type,
                    "source": row.hre,
                    "reference": row.vi,
                    "hypothesis": result["translation"],
                    "model": result["model"],
                    "similarity": result["similarity"],
                    "retrieved_terms": json.dumps(result["retrieved_terms"], ensure_ascii=False),
                    "retrieved_examples": json.dumps(
                        result["retrieved_examples"], ensure_ascii=False
                    ),
                    "replay_processing_ms": result["latency_ms"],
                    "nllb_kaggle_latency_ms": saved.iloc[index].latency_ms,
                }
            )
        predictions = pd.DataFrame(rows)
        predictions.to_csv(output_dir / f"{variant}.csv", index=False, encoding="utf-8-sig")
        scores = corpus_metrics(predictions.hypothesis.astype(str).tolist(), references)
        results.append(
            {
                "model": variant,
                "test_examples": len(test),
                **scores,
                "replay_processing_ms_mean": round(
                    float(np.mean(predictions.replay_processing_ms)), 6
                ),
                "replay_processing_ms_p95": round(
                    float(np.percentile(predictions.replay_processing_ms, 95)), 6
                ),
                "tm_routes": int(predictions.model.eq("translation_memory").sum()),
                "medium_tm_fallbacks": int(
                    predictions.model.eq("retrieval_augmented_tm_fallback").sum()
                ),
                "dictionary_edits": int(
                    predictions.model.eq("retrieval_augmented_dictionary").sum()
                ),
                "latency_note": "Local retrieval/replay only; not end-to-end NLLB latency",
            }
        )
    result = pd.DataFrame(results)
    path = resolve(root, config["evaluation"]["results"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid.yaml")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    config, root = load_config(args.config)
    result = benchmark(root, config)
    if not args.no_mlflow:
        from hre_translate.tracking import log_experiment

        for row in result.to_dict("records"):
            model = str(row["model"])
            log_experiment(
                root=root,
                model_type=model,
                metrics={
                    key: row[key]
                    for key in (
                        "bleu", "chrf_pp", "replay_processing_ms_mean",
                        "replay_processing_ms_p95", "tm_routes", "medium_tm_fallbacks",
                        "dictionary_edits",
                    )
                },
                parameters={**config["retrieval"], "variant": model},
                datasets={
                    "train": resolve(root, config["data"]["train"]),
                    "test": resolve(root, config["data"]["test"]),
                },
                config_path=resolve(root, args.config),
                artifacts=[
                    resolve(root, config["evaluation"]["results"]),
                    resolve(root, config["evaluation"]["predictions_dir"]) / f"{model}.csv",
                ],
                latency_context=(
                    "local_windows_tm_stage4" if model == "translation_memory" else "local_replay"
                ),
                provenance="new_replay_benchmark",
                extra_tags={"neural_generation": "replayed"},
            )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
