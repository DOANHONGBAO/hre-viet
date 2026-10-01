from __future__ import annotations

import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def similarity_document(hre: str, vi: str) -> str:
    """Create a comparison-only form robust to case, spacing, and punctuation changes."""

    def canonical(value: str) -> str:
        return "".join(
            char
            for char in value.casefold()
            if unicodedata.category(char)[0] not in {"C", "P", "Z"}
        )

    return f"{canonical(str(hre))} ⟺ {canonical(str(vi))}"


def near_duplicate_groups(frame: pd.DataFrame, threshold: float = 0.88) -> np.ndarray:
    if len(frame) < 2:
        return np.arange(len(frame), dtype=int)
    documents = [
        similarity_document(record.hre, record.vi)
        for record in frame[["hre", "vi"]].itertuples(index=False)
    ]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(
        documents
    )
    model = NearestNeighbors(metric="cosine", algorithm="brute", radius=1.0 - threshold)
    model.fit(matrix)
    neighbors = model.radius_neighbors(matrix, return_distance=False)
    groups = _UnionFind(len(frame))
    for index, matches in enumerate(neighbors):
        for match in matches:
            if index != int(match):
                groups.union(index, int(match))
    roots = [groups.find(index) for index in range(len(frame))]
    root_to_group = {root: group for group, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([root_to_group[root] for root in roots], dtype=int)


def grouped_split(
    frame: pd.DataFrame,
    *,
    ratios: dict[str, Any],
    seed: int,
    threshold: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    names = ("train", "validation", "test")
    weights = np.asarray([float(ratios[name]) for name in names], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    data = frame.reset_index(drop=True).copy()
    data["near_duplicate_group"] = near_duplicate_groups(data, threshold)
    group_rows: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(data["near_duplicate_group"]):
        group_rows[int(group)].append(index)

    rng = np.random.default_rng(seed)
    groups = list(group_rows)
    tie_breaker = {group: float(rng.random()) for group in groups}
    groups.sort(key=lambda group: (-len(group_rows[group]), tie_breaker[group]))
    target_sizes = weights * len(data)
    current = np.zeros(3, dtype=int)
    assignment: dict[int, str] = {}
    for group in groups:
        size = len(group_rows[group])
        deficit = target_sizes - current
        split_index = int(np.argmax(deficit / np.maximum(target_sizes, 1)))
        assignment[group] = names[split_index]
        current[split_index] += size

    data["split"] = data["near_duplicate_group"].map(assignment)
    result = {name: data.loc[data["split"] == name].reset_index(drop=True) for name in names}
    info = {
        "seed": seed,
        "near_duplicate_threshold": threshold,
        "near_duplicate_groups": len(groups),
        "rows": {name: len(result[name]) for name in names},
        "ratios": {name: round(len(result[name]) / max(len(data), 1), 4) for name in names},
    }
    return result, info


def write_splits(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
