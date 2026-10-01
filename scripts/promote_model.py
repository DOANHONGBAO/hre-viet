"""Review a Candidate run; --apply changes tracking tags only, never deploys."""

from __future__ import annotations

import argparse

from hre_translate.tracking.mlflow_tracking import configure_tracking, promotion_decision
from hre_translate.utils.config import project_root


def main() -> None:
    import mlflow
    from mlflow import MlflowClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--apply", action="store_true", help="Update Champion/Candidate tags")
    parser.add_argument("--config", default="configs/tracking.yaml")
    args = parser.parse_args()
    root = project_root()
    config = configure_tracking(root, args.config)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    candidate = client.get_run(args.run_id)
    if candidate.info.experiment_id != experiment.experiment_id:
        raise ValueError("Candidate is not in the hre-translate experiment")
    if candidate.data.tags.get("management_role") != "Candidate":
        raise ValueError("Run must have Candidate role")
    context = candidate.data.tags.get("latency_context", "")
    if "replay" in context or "kaggle" in context:
        raise ValueError("Replay and cross-hardware runs cannot be promoted")
    if not {"chrf_pp", "latency_ms_mean"} <= candidate.data.metrics.keys():
        raise ValueError("Candidate needs measured chrF++ and end-to-end mean latency")
    champions = client.search_runs(
        [experiment.experiment_id], "tags.management_role = 'Champion'"
    )
    if len(champions) > 1:
        raise ValueError("More than one Champion run; resolve manually")
    if champions:
        champion = champions[0]
        if candidate.data.tags.get("dataset_version") != champion.data.tags.get(
            "dataset_version"
        ):
            raise ValueError("Dataset version differs from Champion")
        if context != champion.data.tags.get("latency_context"):
            raise ValueError("Latency context differs from Champion")
        passed, reason = promotion_decision(
            candidate.data.metrics["chrf_pp"],
            candidate.data.metrics["latency_ms_mean"],
            champion.data.metrics["chrf_pp"],
            champion.data.metrics["latency_ms_mean"],
            **config["promotion"],
        )
    else:
        champion = None
        passed, reason = True, "initial Champion with measured metrics"
    print(f"Candidate {candidate.info.run_id}: {'PASS' if passed else 'REJECT'} — {reason}")
    if not args.apply or not passed:
        return
    if champion:
        client.set_tag(champion.info.run_id, "management_role", "Candidate")
    client.set_tag(candidate.info.run_id, "management_role", "Champion")
    client.set_tag(candidate.info.run_id, "promotion_reason", reason)
    print("Tracking tags updated; no model deployed.")


if __name__ == "__main__":
    main()
