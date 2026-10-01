from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from hre_translate.data.normalize import normalize_hre, normalize_vietnamese
from hre_translate.data.split import grouped_split, write_splits
from hre_translate.utils.config import load_config, resolve


def _clean_column_name(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def read_sheet(workbook: Path, specification: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_excel(
        workbook,
        sheet_name=specification["sheet"],
        header=int(specification["header_row"]) - 1,
        dtype=object,
    )
    frame.columns = [_clean_column_name(column) for column in frame.columns]
    return frame.dropna(how="all")


def _audio_members(archive: Path) -> set[str]:
    if not archive.exists():
        return set()
    with zipfile.ZipFile(archive) as handle:
        return {Path(name).name for name in handle.namelist() if name.lower().endswith(".mp3")}


def prepare_pairs(
    frame: pd.DataFrame,
    specification: dict[str, Any],
    *,
    source_file: str,
    dataset_type: str,
    audio_members: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [specification["hre_column"], specification["vi_column"]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing configured columns in {specification['sheet']}: {missing}")

    output = pd.DataFrame()
    output["hre"] = frame[specification["hre_column"]].map(normalize_hre)
    output["vi"] = frame[specification["vi_column"]].map(normalize_vietnamese)
    id_column = specification.get("id_column")
    output["record_id"] = frame[id_column] if id_column in frame.columns else pd.NA
    notes_column = specification.get("notes_column")
    if notes_column in frame.columns:
        output["notes"] = frame[notes_column].map(normalize_vietnamese)
    output["source_file"] = source_file
    output["source_sheet"] = specification["sheet"]
    output["source_row"] = frame.index + int(specification["header_row"]) + 1
    output["dataset_type"] = dataset_type

    if audio_members is not None:

        def audio_name(value: Any) -> str | None:
            try:
                name = f"{int(float(value))}.mp3"
            except (TypeError, ValueError):
                return None
            return f"TU_VUNG/{name}" if name in audio_members else None

        output["audio_zip_member"] = output["record_id"].map(audio_name)

    before = len(output)
    output = output.dropna(subset=["hre", "vi"])
    empty_removed = before - len(output)
    duplicate_mask = output.duplicated(subset=["hre", "vi"], keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    output = output.loc[~duplicate_mask].reset_index(drop=True)
    return output, {
        "input_rows": before,
        "empty_pairs_removed": empty_removed,
        "exact_duplicates_removed": duplicates_removed,
        "output_rows": len(output),
    }


def run_preprocess(config_path: str | Path | None = None) -> dict[str, Any]:
    config, root = load_config(config_path)
    workbook = resolve(root, config["raw"]["workbook"])
    archive = resolve(root, config["raw"]["audio_archive"])
    processed_dir = resolve(root, config["outputs"]["processed_dir"])
    splits_dir = resolve(root, config["outputs"]["splits_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    raw_lexicon = read_sheet(workbook, config["raw"]["lexicon"])
    raw_sentences = read_sheet(workbook, config["raw"]["sentences"])
    lexicon_all, lexicon_stats = prepare_pairs(
        raw_lexicon,
        config["raw"]["lexicon"],
        source_file=workbook.name,
        dataset_type="lexicon",
    )
    sentences, sentence_stats = prepare_pairs(
        raw_sentences,
        config["raw"]["sentences"],
        source_file=workbook.name,
        dataset_type="sentence",
        audio_members=_audio_members(archive),
    )

    max_tokens = int(config["processing"]["word_pair_max_tokens_per_side"])
    is_word = lexicon_all["hre"].str.split().str.len().le(max_tokens) & lexicon_all[
        "vi"
    ].str.split().str.len().le(max_tokens)
    lexicon = lexicon_all.loc[is_word].copy()
    lexicon["dataset_type"] = "lexicon"
    phrases = lexicon_all.loc[~is_word].copy()
    phrases["dataset_type"] = "phrase"

    outputs = {
        "lexicon": processed_dir / "lexicon.parquet",
        "phrases": processed_dir / "phrases.parquet",
        "sentences": processed_dir / "sentences.parquet",
    }
    lexicon.to_parquet(outputs["lexicon"], index=False)
    phrases.to_parquet(outputs["phrases"], index=False)
    sentences.to_parquet(outputs["sentences"], index=False)

    combined = pd.concat([lexicon, phrases, sentences], ignore_index=True, sort=False)
    split_frames, split_info = grouped_split(
        combined,
        ratios=config["split"],
        seed=int(config["split"]["seed"]),
        threshold=float(config["processing"]["near_duplicate_threshold"]),
    )
    write_splits(split_frames, splits_dir)
    summary = {
        "processed": {
            "lexicon_pairs": len(lexicon),
            "phrase_pairs": len(phrases),
            "sentence_pairs": len(sentences),
            "total_pairs": len(combined),
        },
        "cleaning": {"lexicon_sheet": lexicon_stats, "sentence_sheet": sentence_stats},
        "splits": split_info,
        "outputs": {key: str(value.relative_to(root)) for key, value in outputs.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized H're-Vietnamese datasets.")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run_preprocess(args.config)


if __name__ == "__main__":
    main()
