# HRE-TRANSLATE

Low-resource H’rê–Vietnamese machine translation platform. The long-term project covers data quality, classical baselines, NLLB-based neural translation, retrieval-augmented translation, experiment tracking, serving, deployment, and monitoring.

Stages 1 and 2 are complete: the repository contains the validated data pipeline and classical translation baselines. Neural models, RAG, APIs, containers, Kubernetes, and monitoring remain intentionally out of scope.

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

## Classical baselines

All baselines learn only from `train.parquet` and are evaluated against all 160 held-out rows in `test.parquet`.

### Dictionary and longest phrase matching

The dictionary uses train rows classified as `lexicon` or `phrase`. It first attempts an NFC-normalized exact lookup, then a case-folded lookup, and finally scans left-to-right using the longest available phrase. Unknown words are preserved in the output and returned explicitly.

- Strengths: deterministic, transparent, very fast, and reliable for known terminology.
- Limitations: high OOV rate and no sentence-level reordering or fluency model.

### Translation Memory

Translation Memory indexes train sentences with `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))`. Retrieval returns ranked source, target, cosine similarity, and whether the configurable direct-candidate threshold was met. Its translation baseline uses the top retrieved target, while retaining the top-k list for inspection.

- Strengths: robust to spelling variation and useful when a close sentence already exists.
- Limitations: cannot compose a new translation and is slower than direct dictionary or IBM1 lookup.

### IBM Model 1

IBM Model 1 is implemented in Python with deterministic EM training. It estimates `P(Vietnamese token | H’rê token)` and translates each source token with its most probable compatible target token.

- Strengths: learns lexical correspondences without external aligners and has much lower OOV than the dictionary.
- Limitations: ignores word order and context. A Vietnamese n-gram language model was not added because the current corpus is too small to estimate a reliable general-purpose LM without introducing another smoothing/tuning decision.

### Evaluation

Run any model or all three:

```powershell
python scripts/evaluate.py --model dictionary
python scripts/evaluate.py --model translation_memory
python scripts/evaluate.py --model ibm1
python scripts/evaluate.py --model all
```

Results are written to `artifacts/evaluation/classical_results.csv`; auditable per-example predictions are written under `artifacts/evaluation/predictions/`. BLEU and chrF++ are computed with SacreBLEU. Latency is measured on the local machine and should be treated as environment-specific.

| Model | BLEU | chrF++ | Mean latency (ms) | P95 latency (ms) | OOV rate |
|---|---:|---:|---:|---:|---:|
| Dictionary | 2.330843 | 11.724265 | 0.031586 | 0.057650 | 0.770661 |
| Translation Memory | 4.552510 | 17.809439 | 34.052159 | 35.381730 | N/A |
| IBM Model 1 | 4.249287 | 20.718799 | 0.260700 | 0.821305 | 0.185950 |

The low absolute scores are genuine held-out results and are expected for these non-neural baselines on a small, heterogeneous corpus.

## Roadmap

1. **Stage 1 — complete:** data inspection, normalization, validation, leakage-aware splitting, and tests.
2. **Stage 2 — complete:** dictionary, longest phrase, Translation Memory, IBM Model 1, and shared evaluation.
3. NLLB neural machine translation and experiment tracking.
4. Retrieval-augmented translation.
5. FastAPI and web UI.
6. Docker, CI/CD, Kubernetes, and monitoring.
