"""Idempotently import existing Stage 2–4 result files as historical MLflow runs."""

from __future__ import annotations

import argparse

import pandas as pd

from hre_translate.tracking.mlflow_tracking import (
    configure_tracking,
    file_sha256,
    log_experiment,
)
from hre_translate.utils.config import load_config, project_root, resolve


def main() -> None:
    import mlflow
    from mlflow import MlflowClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tracking.yaml")
    args = parser.parse_args()
    root = project_root()
    tracking = configure_tracking(root, args.config)
    experiment = mlflow.get_experiment_by_name(tracking["experiment_name"])
    client = MlflowClient()
    existing = {
        run.data.tags.get("source_record_id"): run
        for run in client.search_runs([experiment.experiment_id], max_results=1000)
        if run.data.tags.get("source_record_id")
    }
    specs = [
        ("classical", root / "artifacts/evaluation/classical_results.csv"),
        ("nllb", root / "artifacts/evaluation/nllb_results.csv"),
        ("hybrid", root / "artifacts/evaluation/stage4_results.csv"),
    ]
    for stage, result_path in specs:
        if not result_path.exists():
            continue
        config_name = {
            "classical": "configs/classical.yaml",
            "nllb": "configs/nllb.yaml",
            "hybrid": "configs/hybrid.yaml",
        }[stage]
        config, _ = load_config(config_name)
        train = resolve(root, config["data"]["train"])
        test = resolve(root, config["data"]["test"])
        for row in pd.read_csv(result_path).to_dict("records"):
            model = str(row["model"])
            record_id = f"{stage}:{model}:{file_sha256(result_path)}"
            if record_id in existing:
                if stage == "nllb":
                    for section in ("model", "lora", "generation"):
                        for key, value in config[section].items():
                            param_key = f"{section}.{key}"
                            if param_key not in existing[record_id].data.params:
                                client.log_param(
                                    existing[record_id].info.run_id, param_key, str(value)
                                )
                continue
            predictions = (
                root / "artifacts/evaluation/predictions" / f"{model}.csv"
                if stage == "classical"
                else root / "artifacts/evaluation/predictions/nllb_lora.csv"
                if stage == "nllb"
                else root / "artifacts/evaluation/stage4_predictions" / f"{model}.csv"
            )
            metric_keys = (
                "bleu", "chrf_pp", "latency_ms_mean", "latency_ms_p95",
                "replay_processing_ms_mean", "replay_processing_ms_p95", "oov_rate",
            )
            metrics = {key: row[key] for key in metric_keys if key in row}
            context = (
                "local_windows_classical" if stage == "classical" else
                "kaggle_t4" if stage == "nllb" else
                "local_windows_tm_stage4" if model == "translation_memory" else "local_replay"
            )
            if stage == "hybrid" and model == "translation_memory":
                metrics["latency_ms_mean"] = row["replay_processing_ms_mean"]
                metrics["latency_ms_p95"] = row["replay_processing_ms_p95"]
            params = config if stage == "nllb" else config.get(
                model, config.get("retrieval", {})
            )
            artifacts = [result_path, predictions]
            if stage == "nllb":
                artifacts.extend(
                    [
                        root / "artifacts/evaluation/nllb_training_summary.json",
                        root / "artifacts/evaluation/nllb_provenance.json",
                        root / "artifacts/models/nllb/final_adapter/adapter_config.json",
                        root / "artifacts/models/nllb/final_adapter/adapter_model.safetensors",
                    ]
                )
            run_id = log_experiment(
                root=root,
                model_type=model,
                metrics=metrics,
                parameters=params,
                datasets={"train": train, "test": test},
                config_path=root / config_name,
                artifacts=artifacts,
                latency_context=context,
                provenance="historical_import",
                tracking_config=args.config,
                extra_tags={
                    "source_record_id": record_id,
                    "git_commit_scope": "import_time_not_original_run",
                    "neural_generation": "saved_kaggle" if stage != "classical" else "none",
                },
            )
            existing[record_id] = client.get_run(run_id)
            print(f"Imported {stage}/{model}: {run_id}")


if __name__ == "__main__":
    main()
