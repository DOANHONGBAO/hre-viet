"""MLflow tracking and manual Champion/Candidate review helpers."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from hre_translate.utils.config import load_config, resolve


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> tuple[str, bool]:
    command = ["git", "-c", f"safe.directory={root.as_posix()}"]
    try:
        commit = subprocess.run(
            [*command, "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [*command, "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def _flatten(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        output: dict[str, str] = {}
        for key, child in value.items():
            output.update(_flatten(f"{prefix}.{key}" if prefix else str(key), child))
        return output
    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False)}
    return {prefix: str(value)}


def configure_tracking(root: Path, config_path: str | Path = "configs/tracking.yaml") -> dict:
    import mlflow

    config, _ = load_config(config_path)
    store = resolve(root, config["store"])
    store.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = resolve(root, config["artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")
    name = str(config["experiment_name"])
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(name)
    return config


def log_experiment(
    *,
    root: Path,
    model_type: str,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    datasets: dict[str, Path],
    config_path: Path,
    artifacts: list[Path],
    latency_context: str,
    provenance: str = "new_run",
    tracking_config: str | Path = "configs/tracking.yaml",
    extra_tags: dict[str, str] | None = None,
) -> str:
    import mlflow

    configure_tracking(root, tracking_config)
    commit, dirty = git_state(root)
    dataset_hashes = {name: file_sha256(path) for name, path in datasets.items()}
    with mlflow.start_run(run_name=f"{model_type}-{provenance}") as run:
        mlflow.set_tags(
            {
                "model_type": model_type,
                "git_commit": commit,
                "git_dirty": str(dirty).lower(),
                "dataset_version": hashlib.sha256(
                    json.dumps(dataset_hashes, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "latency_context": latency_context,
                "provenance": provenance,
                "management_role": "Candidate",
                **(extra_tags or {}),
            }
        )
        mlflow.log_params(_flatten("", parameters))
        for key, value in dataset_hashes.items():
            mlflow.set_tag(f"dataset_sha256.{key}", value)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(config_path), artifact_path="config")
        for path in artifacts:
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="outputs")
        mlflow.log_dict(
            {"dataset_hashes": dataset_hashes, "git_commit": commit, "git_dirty": dirty},
            "provenance.json",
        )
        return run.info.run_id


def promotion_decision(
    candidate_chrf: float,
    candidate_latency: float,
    champion_chrf: float,
    champion_latency: float,
    *,
    max_relative_latency: float = 1.5,
    max_absolute_latency_increase_ms: float = 5.0,
) -> tuple[bool, str]:
    values = [candidate_chrf, candidate_latency, champion_chrf, champion_latency]
    if not all(math.isfinite(value) for value in values) or min(
        candidate_latency, champion_latency
    ) < 0:
        return False, "invalid or missing metric"
    if candidate_chrf <= champion_chrf:
        return False, "chrF++ did not improve"
    allowed = max(
        champion_latency * max_relative_latency,
        champion_latency + max_absolute_latency_increase_ms,
    )
    if candidate_latency > allowed:
        return False, "latency increase exceeds threshold"
    return True, "chrF++ improved and latency is within threshold"
