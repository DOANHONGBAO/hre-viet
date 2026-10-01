# HRE-TRANSLATE

Low-resource H’rê–Vietnamese machine translation platform. The long-term project covers data quality, classical baselines, NLLB-based neural translation, retrieval-augmented translation, experiment tracking, serving, deployment, and monitoring.

The repository is currently at **Stage 1: project foundation and data pipeline**. Neural models, RAG, APIs, containers, Kubernetes, and monitoring are intentionally out of scope.

## Current data

The original files are kept unchanged in `data/`:

- An Excel workbook with sheets `TỪ VỰNG` and `CÂU`.
- A ZIP archive containing numbered MP3 recordings. Audio is referenced as sentence metadata and is not extracted.

Run the inspection command for authoritative counts and detailed null, duplicate, length, vocabulary, character, Unicode, and near-duplicate statistics. The latest generated summary is also available in `artifacts/data_report.md`.

The current raw inspection found 1,000 usable entries in `TỪ VỰNG` and 601 usable entries in `CÂU`. After normalization and exact deduplication, the processed corpus contains 194 word pairs, 802 phrase pairs, and 599 sentence pairs (1,595 total). The audio ZIP contains 1,001 MP3 files numbered 1–1001.

## Setup

The pre-existing `hre-translate` environment on this machine was found to be internally inconsistent, so it was preserved and an isolated Stage 1 environment was created:

```powershell
conda activate hre-translate-stage1
python -m pip install -e ".[dev]"
```

For a clean machine, the equivalent setup is:

```powershell
conda create -n hre-translate-stage1 python=3.11 -y
conda activate hre-translate-stage1
python -m pip install -e ".[dev]"
```

## Reproduce Stage 1

Run from the repository root after activating the environment:

```powershell
python -m hre_translate.data.inspect
python -m hre_translate.data.preprocess
python -m hre_translate.data.validate
pytest
ruff check src tests
```

Inspection writes JSON and Markdown reports under `artifacts/`. Preprocessing writes normalized datasets under `data/processed/`, removes empty and exact duplicate pairs, identifies near-duplicate groups with character n-gram TF-IDF, and writes group-aware 80/10/10 splits under `data/splits/`. Validation exits non-zero on failure.

The generated group-aware split currently contains 1,275 train, 160 validation, and 160 test pairs. Validation reports no exact duplicate or severe near-duplicate leakage across these splits.

## Roadmap

1. **Stage 1 — current:** data inspection, normalization, validation, leakage-aware splitting, and tests.
2. Classical translation baselines.
3. NLLB neural machine translation and experiment tracking.
4. Retrieval-augmented translation.
5. FastAPI and web UI.
6. Docker, CI/CD, Kubernetes, and monitoring.
