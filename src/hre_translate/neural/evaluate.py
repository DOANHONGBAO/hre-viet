from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hre_translate.neural.config import NLLBConfig
from hre_translate.neural.data import load_parallel_frame
from hre_translate.neural.inference import NLLBTranslator
from hre_translate.neural.metrics import corpus_metrics
from hre_translate.neural.runtime import detect_runtime


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def evaluate_adapter(config: NLLBConfig, root: Path, adapter_path: Path) -> pd.DataFrame:
    runtime = detect_runtime(config.training.fp16)
    translator = NLLBTranslator.from_adapter(
        adapter_path,
        model_name=config.model.name,
        source_language=config.model.source_language_proxy,
        target_language=config.model.target_language,
        device=runtime.device,
        beam_size=config.generation.beam_size,
        max_source_length=config.training.max_source_length,
        max_new_tokens=config.generation.max_new_tokens,
    )
    test = load_parallel_frame(_resolve(root, config.data.test))
    if not test.empty:
        translator.translate(str(test.iloc[0]["hre"]))  # unmeasured warm-up

    predictions = []
    hypotheses = []
    references = []
    latencies = []
    for row in test.itertuples(index=False):
        result = translator.translate(str(row.hre))
        hypotheses.append(result["translation"])
        references.append(str(row.vi))
        latencies.append(float(result["latency_ms"]))
        predictions.append(
            {
                "dataset_type": row.dataset_type,
                "source": row.hre,
                "reference": row.vi,
                "hypothesis": result["translation"],
                "latency_ms": result["latency_ms"],
            }
        )

    scores = corpus_metrics(hypotheses, references)
    result = pd.DataFrame(
        [
            {
                "model": "nllb_lora",
                "evaluated_at_utc": datetime.now(UTC).isoformat(),
                "train_examples": len(load_parallel_frame(_resolve(root, config.data.train))),
                "test_examples": len(test),
                "examples": len(test),
                **scores,
                "latency_ms_mean": round(float(np.mean(latencies)), 6),
                "latency_ms_p95": round(float(np.percentile(latencies, 95)), 6),
                "oov_rate": np.nan,
            }
        ]
    )
    results_path = _resolve(root, config.evaluation.results)
    predictions_path = _resolve(root, config.evaluation.predictions)
    comparison_path = _resolve(root, config.evaluation.comparison)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(results_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    classical_path = _resolve(root, config.evaluation.classical_results)
    frames = [pd.read_csv(classical_path), result] if classical_path.exists() else [result]
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )
    return result
