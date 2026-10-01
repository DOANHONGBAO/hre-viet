from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from hre_translate.models.dictionary import DictionaryTranslator
from hre_translate.models.retrieval import TranslationMemory
from hre_translate.models.statistical import IBMModel1
from hre_translate.utils.config import load_config, resolve


class TranslationModel(Protocol):
    def translate(self, text: str) -> dict[str, Any]: ...


def evaluate_model(
    model: TranslationModel,
    test_frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    hypotheses: list[str] = []
    references: list[str] = []
    latencies_ms: list[float] = []
    unknown_count = 0
    source_token_count = 0
    oov_applicable = False
    prediction_rows = []

    for row in test_frame[["hre", "vi", "dataset_type"]].itertuples(index=False):
        start = time.perf_counter()
        result = model.translate(str(row.hre))
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        hypothesis = str(result.get("translation", ""))
        reference = str(row.vi)
        hypotheses.append(hypothesis)
        references.append(reference)
        if "unknown_terms" in result:
            oov_applicable = True
            unknown_count += len(result.get("unknown_terms", []))
        source_token_count += len(str(row.hre).split())
        prediction_rows.append(
            {
                "dataset_type": row.dataset_type,
                "source": row.hre,
                "reference": reference,
                "hypothesis": hypothesis,
                "details": json.dumps(result, ensure_ascii=False),
            }
        )

    bleu = BLEU(effective_order=True).corpus_score(hypotheses, [references]).score
    chrf_pp = CHRF(word_order=2).corpus_score(hypotheses, [references]).score
    metrics = {
        "examples": len(test_frame),
        "bleu": round(float(bleu), 6),
        "chrf_pp": round(float(chrf_pp), 6),
        "latency_ms_mean": round(float(np.mean(latencies_ms)), 6),
        "latency_ms_p95": round(float(np.percentile(latencies_ms, 95)), 6),
        "oov_rate": (
            round(unknown_count / source_token_count, 6)
            if oov_applicable and source_token_count
            else np.nan
        ),
    }
    return metrics, pd.DataFrame(prediction_rows)


def _select_types(frame: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    return frame.loc[frame["dataset_type"].isin(types)].reset_index(drop=True)


def build_model(name: str, train: pd.DataFrame, config: dict[str, Any]) -> TranslationModel:
    if name == "dictionary":
        options = config["dictionary"]
        training = _select_types(train, options["training_types"])
        return DictionaryTranslator.from_frame(
            training, preserve_unknown=bool(options["preserve_unknown"])
        )
    if name == "translation_memory":
        options = config["translation_memory"]
        training = _select_types(train, options["training_types"])
        return TranslationMemory(
            ngram_range=tuple(int(value) for value in options["ngram_range"]),
            direct_match_threshold=float(options["direct_match_threshold"]),
        ).fit(training)
    if name == "ibm1":
        options = config["ibm1"]
        training = _select_types(train, options["training_types"])
        return IBMModel1.from_frame(
            training,
            iterations=int(options["iterations"]),
            null_token=str(options["null_token"]),
        )
    raise ValueError(f"Unsupported model: {name}")


def training_types_for(name: str, config: dict[str, Any]) -> list[str]:
    section = "translation_memory" if name == "translation_memory" else name
    return list(config[section]["training_types"])


def run_evaluation(
    model_names: list[str], config_path: str | Path | None = None, *, track: bool = False
) -> pd.DataFrame:
    chosen_config = config_path or root_classical_config()
    config, root = load_config(chosen_config)
    config_file = resolve(root, str(chosen_config))
    train_path = resolve(root, config["data"]["train"])
    test_path = resolve(root, config["data"]["test"])
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    output_path = resolve(root, config["evaluation"]["output"])
    predictions_dir = resolve(root, config["evaluation"]["predictions_dir"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    result_rows = []
    for name in model_names:
        training_examples = len(_select_types(train, training_types_for(name, config)))
        training_start = time.perf_counter()
        model = build_model(name, train, config)
        training_seconds = time.perf_counter() - training_start
        metrics, predictions = evaluate_model(model, test)
        predictions.to_csv(predictions_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        result_rows.append(
            {
                "model": name,
                "evaluated_at_utc": datetime.now(UTC).isoformat(),
                "train_examples": training_examples,
                "test_examples": len(test),
                "training_seconds": round(training_seconds, 6),
                **metrics,
            }
        )

    current = pd.DataFrame(result_rows)
    if not existing.empty and "model" in existing.columns:
        existing = existing.loc[~existing["model"].isin(model_names)]
    combined = pd.concat([existing, current], ignore_index=True).sort_values("model")
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    if track:
        from hre_translate.tracking import log_experiment

        for row in result_rows:
            name = str(row["model"])
            log_experiment(
                root=root,
                model_type=name,
                metrics={
                    key: row[key]
                    for key in ("bleu", "chrf_pp", "latency_ms_mean", "latency_ms_p95", "oov_rate")
                },
                parameters=config[name],
                datasets={"train": train_path, "test": test_path},
                config_path=config_file,
                artifacts=[output_path, predictions_dir / f"{name}.csv"],
                latency_context="local_windows_classical",
            )
    return current.sort_values("model").reset_index(drop=True)


def root_classical_config() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "classical.yaml"
