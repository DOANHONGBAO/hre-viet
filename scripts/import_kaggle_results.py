"""Verify a Kaggle Stage 3 archive before importing its final adapter and metrics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from hre_translate.neural.metrics import corpus_metrics
from hre_translate.utils.config import project_root

PREFIX = "artifacts/"
PREDICTIONS = PREFIX + "evaluation/predictions/nllb_lora.csv"
RESULTS = PREFIX + "evaluation/nllb_results.csv"
COMPARISON = PREFIX + "evaluation/model_comparison.csv"
TRAINING = PREFIX + "models/nllb/training_summary.json"
ADAPTER_PREFIX = PREFIX + "models/nllb/final_adapter/"
FINAL_ADAPTER = ADAPTER_PREFIX + "adapter_model.safetensors"


def _sha256(stream: io.BufferedIOBase) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as handle:
        return pd.read_csv(handle)


def _check_archive_names(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Archive contains duplicate entry names")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive entry: {name}")


def verify_archive(archive: zipfile.ZipFile, root: Path) -> dict[str, object]:
    _check_archive_names(archive)
    states = sorted(
        (
            name
            for name in archive.namelist()
            if name.startswith(PREFIX + "models/nllb/checkpoint-")
            and name.endswith("/trainer_state.json")
        ),
        key=lambda name: int(name.split("checkpoint-")[1].split("/")[0]),
    )
    if not states:
        raise ValueError("No trainer state found in Kaggle archive")
    state_name = states[-1]
    required = {PREDICTIONS, RESULTS, COMPARISON, TRAINING, state_name, FINAL_ADAPTER}
    missing = required - set(archive.namelist())
    if missing:
        raise ValueError(f"Missing required Kaggle artifacts: {sorted(missing)}")
    bad_file = archive.testzip()
    if bad_file:
        raise ValueError(f"ZIP CRC verification failed: {bad_file}")

    predictions = _read_csv(archive, PREDICTIONS)
    results = _read_csv(archive, RESULTS)
    comparison = _read_csv(archive, COMPARISON)
    test = pd.read_parquet(root / "data" / "splits" / "test.parquet")
    columns = {"dataset_type", "source", "reference", "hypothesis", "latency_ms"}
    if not columns <= set(predictions.columns):
        missing_columns = sorted(columns - set(predictions.columns))
        raise ValueError(f"Prediction columns missing: {missing_columns}")
    if len(predictions) != len(test) or len(results) != 1:
        raise ValueError("Prediction/result row count does not match the fixed test split")
    for actual, expected in (
        (predictions["source"], test["hre"]),
        (predictions["reference"], test["vi"]),
        (predictions["dataset_type"], test["dataset_type"]),
    ):
        if actual.astype(str).tolist() != expected.astype(str).tolist():
            raise ValueError("Kaggle predictions do not match the fixed Stage 1 test split")
    if predictions[["source", "reference", "hypothesis", "latency_ms"]].isna().any().any():
        raise ValueError("Predictions contain missing values")
    latencies = pd.to_numeric(predictions["latency_ms"], errors="raise").to_numpy()
    if not np.isfinite(latencies).all() or (latencies < 0).any():
        raise ValueError("Predictions contain invalid latencies")
    scores = corpus_metrics(predictions["hypothesis"].tolist(), predictions["reference"].tolist())
    result = results.iloc[0]
    if result["model"] != "nllb_lora" or int(result["test_examples"]) != len(test):
        raise ValueError("NLLB result metadata does not match the test split")
    for key, value in {
        **scores,
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }.items():
        if not np.isclose(float(result[key]), value, atol=1e-5, rtol=0):
            raise ValueError(f"Reported {key} does not match predictions: {result[key]} vs {value}")
    classical = pd.read_csv(root / "artifacts" / "evaluation" / "classical_results.csv")
    expected_comparison = pd.concat([classical, results], ignore_index=True, sort=False)
    if comparison["model"].tolist() != expected_comparison["model"].tolist():
        raise ValueError("Comparison model list differs from local classical results")
    for model in classical["model"]:
        old = classical.loc[classical["model"] == model].iloc[0]
        reported = comparison.loc[comparison["model"] == model].iloc[0]
        for metric in ("bleu", "chrf_pp"):
            if not np.isclose(float(old[metric]), float(reported[metric]), atol=1e-6):
                raise ValueError(f"Comparison changed Stage 2 {model} {metric}")

    training = json.loads(archive.read(TRAINING))
    state = json.loads(archive.read(state_name))
    if training["base_model"] != "facebook/nllb-200-distilled-600M":
        raise ValueError("Unexpected base model")
    train_count = len(pd.read_parquet(root / "data" / "splits" / "train.parquet"))
    validation_count = len(pd.read_parquet(root / "data" / "splits" / "validation.parquet"))
    if (
        int(training["train_rows"]) != train_count
        or int(training["validation_rows"]) != validation_count
    ):
        raise ValueError("Unexpected train/validation counts")
    if not np.isclose(
        float(training["training_metrics"]["epoch"]), float(state["num_train_epochs"])
    ) or int(state["global_step"]) != int(state_name.split("checkpoint-")[1].split("/")[0]):
        raise ValueError("Training summary and last checkpoint disagree")
    with archive.open(FINAL_ADAPTER) as handle:
        final_hash = _sha256(handle)
    best_step = int(state["best_global_step"])
    best_adapter = PREFIX + f"models/nllb/checkpoint-{best_step}/adapter_model.safetensors"
    best_hash = None
    if best_adapter in archive.namelist():
        with archive.open(best_adapter) as handle:
            best_hash = _sha256(handle)
        if final_hash != best_hash:
            raise ValueError("Final adapter does not match the selected best checkpoint")
    return {
        "test_rows": len(test),
        "scores": scores,
        "latency_ms_mean": round(float(np.mean(latencies)), 6),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 6),
        "training_epochs": float(training["training_metrics"]["epoch"]),
        "best_step": best_step,
        "best_validation_chrf_pp": float(state["best_metric"]),
        "final_adapter_sha256": final_hash,
        "best_adapter_sha256": best_hash,
    }


def import_archive(path: Path, root: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        verified = verify_archive(archive, root)
        selected = [PREDICTIONS, RESULTS, COMPARISON]
        selected.extend(
            name
            for name in archive.namelist()
            if name.startswith(ADAPTER_PREFIX) and not name.endswith("/")
        )
        for name in selected:
            destination = root / Path(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        evaluation = root / "artifacts" / "evaluation"
        (evaluation / "nllb_training_summary.json").write_bytes(archive.read(TRAINING))
        state_name = max(
            (
                name
                for name in archive.namelist()
                if name.startswith(PREFIX + "models/nllb/checkpoint-")
                and name.endswith("/trainer_state.json")
            ),
            key=lambda name: int(name.split("checkpoint-")[1].split("/")[0]),
        )
        (evaluation / "nllb_training_state.json").write_bytes(archive.read(state_name))
    with path.open("rb") as handle:
        verified["source_archive_sha256"] = _sha256(handle)
    provenance_path = root / "artifacts" / "evaluation" / "nllb_provenance.json"
    provenance_path.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    verified = import_archive(args.archive, project_root())
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
