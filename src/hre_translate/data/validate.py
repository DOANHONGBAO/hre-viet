from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from hre_translate.data.split import similarity_document
from hre_translate.utils.config import load_config, resolve


def validate_frame(frame: pd.DataFrame, *, max_characters: int = 1000) -> list[str]:
    errors: list[str] = []
    missing_columns = {"hre", "vi"} - set(frame.columns)
    if missing_columns:
        return [f"missing required columns: {sorted(missing_columns)}"]
    if frame[["hre", "vi"]].isna().any().any():
        errors.append("NaN found in hre or vi")
    for column in ("hre", "vi"):
        text = frame[column].fillna("").astype(str)
        if text.str.strip().eq("").any():
            errors.append(f"empty values found in {column}")
        if text.str.len().gt(max_characters).any():
            errors.append(f"values longer than {max_characters} characters found in {column}")
        try:
            for value in text:
                value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            errors.append(f"invalid UTF-8-compatible text in {column}: {exc}")
    if frame.duplicated(["hre", "vi"]).any():
        errors.append("exact duplicate pairs found")
    return errors


def severe_cross_split_leakage(
    splits: dict[str, pd.DataFrame], threshold: float
) -> list[dict[str, Any]]:
    rows = []
    for split, frame in splits.items():
        for record in frame[["hre", "vi"]].itertuples(index=False):
            rows.append((split, str(record.hre), str(record.vi)))
    if len(rows) < 2:
        return []
    documents = [similarity_document(hre, vi) for _, hre, vi in rows]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(documents)
    model = NearestNeighbors(metric="cosine", algorithm="brute", radius=1.0 - threshold).fit(matrix)
    distances, neighbors = model.radius_neighbors(matrix, return_distance=True)
    leaks = []
    for left, (row_distances, row_neighbors) in enumerate(zip(distances, neighbors, strict=True)):
        for distance, right_value in zip(row_distances, row_neighbors, strict=True):
            right = int(right_value)
            if right <= left or rows[left][0] == rows[right][0]:
                continue
            leaks.append(
                {
                    "left_split": rows[left][0],
                    "right_split": rows[right][0],
                    "similarity": round(1.0 - float(distance), 4),
                    "left_hre": rows[left][1],
                    "right_hre": rows[right][1],
                }
            )
    return sorted(leaks, key=lambda item: item["similarity"], reverse=True)


def run_validation(config_path: str | Path | None = None) -> dict[str, Any]:
    config, root = load_config(config_path)
    processed_dir = resolve(root, config["outputs"]["processed_dir"])
    splits_dir = resolve(root, config["outputs"]["splits_dir"])
    max_characters = int(config["processing"]["max_reasonable_characters"])
    errors: dict[str, list[str]] = {}

    for name in ("lexicon", "phrases", "sentences"):
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            errors[str(path.relative_to(root))] = ["file does not exist"]
            continue
        frame_errors = validate_frame(pd.read_parquet(path), max_characters=max_characters)
        if frame_errors:
            errors[str(path.relative_to(root))] = frame_errors

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        path = splits_dir / f"{name}.parquet"
        if not path.exists():
            errors[str(path.relative_to(root))] = ["file does not exist"]
            continue
        splits[name] = pd.read_parquet(path)
        frame_errors = validate_frame(splits[name], max_characters=max_characters)
        if frame_errors:
            errors[str(path.relative_to(root))] = frame_errors

    if len(splits) == 3:
        combined = pd.concat(
            [frame.assign(_split=name) for name, frame in splits.items()], ignore_index=True
        )
        duplicated = combined.duplicated(["hre", "vi"], keep=False)
        crosses_splits = (
            combined.loc[duplicated].groupby(["hre", "vi"])["_split"].nunique().gt(1).any()
        )
        if duplicated.any() and crosses_splits:
            errors["cross_split"] = ["exact duplicate pairs occur across splits"]
        leaks = severe_cross_split_leakage(
            splits, float(config["processing"]["severe_leakage_threshold"])
        )
        if leaks:
            errors.setdefault("cross_split", []).append(
                f"{len(leaks)} severe near-duplicate pair(s) occur across splits"
            )
    else:
        leaks = []

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "severe_cross_split_leakage_examples": leaks[:20],
        "split_rows": {name: len(frame) for name, frame in splits.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed datasets and splits.")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = run_validation(args.config)
    if result["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
