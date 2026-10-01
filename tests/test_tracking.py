from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from hre_translate.tracking.mlflow_tracking import log_experiment, promotion_decision


@pytest.mark.parametrize(
    ("candidate_chrf", "candidate_latency", "expected"),
    [(22.0, 12.0, True), (20.0, 10.0, False), (22.0, 16.0, False)],
)
def test_promotion_rule(candidate_chrf, candidate_latency, expected):
    passed, _ = promotion_decision(
        candidate_chrf, candidate_latency, 20.0, 10.0,
        max_relative_latency=1.5, max_absolute_latency_increase_ms=5.0,
    )
    assert passed is expected


def test_mlflow_logs_traceable_run(tmp_path: Path):
    config = tmp_path / "tracking.yaml"
    config.write_text(
        "experiment_name: hre-translate\nstore: mlflow.db\nartifacts: mlartifacts\n",
        encoding="utf-8",
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text("seed: 42\n", encoding="utf-8")
    train = tmp_path / "train.txt"
    train.write_text("training pair", encoding="utf-8")
    test = tmp_path / "test.txt"
    test.write_text("test pair", encoding="utf-8")
    output = tmp_path / "predictions.csv"
    output.write_text("source,hypothesis\na,b\n", encoding="utf-8")
    run_id = log_experiment(
        root=tmp_path,
        model_type="dictionary",
        metrics={"bleu": 1.2, "chrf_pp": 3.4, "latency_ms_mean": 0.2},
        parameters={"iterations": 1},
        datasets={"train": train, "test": test},
        config_path=model_config,
        artifacts=[output],
        latency_context="test_machine",
        tracking_config=config,
    )
    run = MlflowClient().get_run(run_id)
    assert run.data.metrics["chrf_pp"] == 3.4
    assert run.data.params["iterations"] == "1"
    assert run.data.tags["model_type"] == "dictionary"
    assert len(run.data.tags["dataset_version"]) == 64
    assert run.data.tags["management_role"] == "Candidate"
    assert mlflow.get_experiment(run.info.experiment_id).name == "hre-translate"
