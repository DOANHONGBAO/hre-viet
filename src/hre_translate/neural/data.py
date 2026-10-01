from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_parallel_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"hre", "vi"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing parallel columns: {sorted(missing)}")
    frame = frame.dropna(subset=["hre", "vi"]).copy()
    frame["hre"] = frame["hre"].astype(str)
    frame["vi"] = frame["vi"].astype(str)
    return frame.reset_index(drop=True)


def parallel_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"source": str(record.hre), "target": str(record.vi)}
        for record in frame[["hre", "vi"]].itertuples(index=False)
    ]


def tokenize_batch(
    examples: dict[str, list[str]],
    tokenizer: Any,
    *,
    source_language: str,
    target_language: str,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, Any]:
    tokenizer.src_lang = source_language
    tokenizer.tgt_lang = target_language
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["target"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
