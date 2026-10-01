# HRE-TRANSLATE

Low-resource H’rê–Vietnamese machine translation platform. The long-term project covers data quality, classical baselines, NLLB-based neural translation, retrieval-augmented translation, experiment tracking, serving, deployment, and monitoring.

Stages 1–5 are complete. Stage 4 combines the saved NLLB/LoRA run with train-only lexical and sentence retrieval; Stage 5 tracks experiments locally with MLflow. APIs, containers, Kubernetes, and monitoring remain intentionally out of scope.

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
python -m pip install -e ".[tracking]"
```

For a clean machine, the equivalent setup is:

```powershell
conda create -n hre-translate-stage1 python=3.11 -y
conda activate hre-translate-stage1
python -m pip install -e ".[dev]"
python -m pip install -e ".[tracking]"
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

## NLLB + LoRA

Stage 3 fine-tunes `facebook/nllb-200-distilled-600M` with LoRA adapters on `q_proj` and `v_proj`; the frozen base weights are not fully fine-tuned. The fixed Stage 1 train/validation/test split remains unchanged.

H’rê has no official NLLB-200 language code. The configurable initial strategy uses `vie_Latn` as a source proxy because both languages use Latin orthography with Vietnamese-style diacritics, while generation is forced to the official Vietnamese target token `vie_Latn`. This is a pragmatic initialization heuristic, not a claim that H’rê is Vietnamese. Details and alternatives are recorded in `DECISIONS.md`.

Install and verify the neural environment:

```powershell
python -m pip install -e ".[dev,neural,tracking]"
python scripts/check_neural_setup.py
```

Train and automatically evaluate the best adapter when a suitable CUDA GPU is available; the second command is only for a separate re-evaluation:

```powershell
python scripts/train_nllb.py --config configs/nllb.yaml
python scripts/evaluate_nllb.py --config configs/nllb.yaml `
  --adapter artifacts/models/nllb/final_adapter
```

Training writes adapter-only checkpoints to `artifacts/models/nllb/`. Evaluation writes real BLEU, chrF++, latency, predictions, and a comparison with Stage 2 under `artifacts/evaluation/`.

The completed Kaggle run trained for 10 epochs (400 optimizer steps) on a Tesla T4. The highest validation chrF++ was 4.800153 at step 40 (epoch 1). The final adapter is byte-identical to that best checkpoint, because the trainer restored the best model before saving. Its 1,179,648 trainable parameters are 0.191423% of the 616,253,440 total parameters. Training took 2,028.66 seconds, about 33.8 minutes.

| Model | Test BLEU | Test chrF++ | Mean latency (ms) |
|---|---:|---:|---:|
| Dictionary | 2.330843 | 11.724265 | 0.031586 |
| IBM Model 1 | 4.249287 | 20.718799 | 0.260700 |
| Translation Memory | 4.552510 | 17.809439 | 34.052159 |
| NLLB + LoRA | 0.435135 | 5.879872 | 273.075916 |

The NLLB scores were independently recomputed from all 160 saved predictions and checked against the unchanged test split. The model currently underperforms every Stage 2 baseline. Sample outputs include generic Vietnamese phrases and copied H’rê tokens; this experiment is a reproducible negative result, not a deployable translation model. Latencies were measured on different hardware (Stage 2 locally, NLLB on Kaggle T4) and should not be interpreted as a controlled speed comparison. See [comparison](artifacts/evaluation/model_comparison.csv), [per-example predictions](artifacts/evaluation/predictions/nllb_lora.csv), and [provenance](artifacts/evaluation/nllb_provenance.json).

The Kaggle archive was verified and imported with:

```powershell
python scripts/import_kaggle_results.py artifacts/kaggle/hre-nllb-stage3-output.zip
```

The ZIP remains in `artifacts/kaggle/`; the selected adapter is available locally in `artifacts/models/nllb/final_adapter/`. The archive did not include the modified Kaggle `configs/nllb.yaml`, so all run settings beyond those recorded in the training summary and checkpoint state cannot be independently recovered. The repository config now defaults to `gradient_checkpointing: false`, the setting used to resolve the Kaggle gradient error.

The local machine exposes a GTX 1060 with 3 GB VRAM, but the installed PyTorch build is CPU-only and that VRAM capacity is not sufficient for stable NLLB-600M fine-tuning. Use the prepared [Kaggle instructions](kaggle/README_KAGGLE.md) and package:

```powershell
python scripts/package_kaggle.py
```

Upload `artifacts/kaggle/hre-translate-stage3-kaggle.zip` as a private Kaggle Dataset, attach it to a GPU notebook, and follow the included commands.

## Retrieval-augmented translation (Stage 4)

The hybrid service normalizes H’rê text, retrieves longest-match terms from the train lexicon, and retrieves top-k train sentences by `char_wb` TF-IDF cosine similarity. Its structured result contains `translation`, `model`, `retrieved_terms`, `retrieved_examples`, `similarity`, and `latency_ms`. Thresholds and retrieval settings are in `configs/hybrid.yaml` (medium 0.50, high 0.85 by default).

Routing is reference-free: high similarity uses the TM target directly; medium similarity uses NLLB with retrieval context, falling back to the TM target only when at least half of the source word types (length ≥3) are copied in the neural output. Otherwise, matched dictionary terms copied verbatim into the neural output are replaced with their Vietnamese targets using longest-match order. Low similarity uses NLLB unchanged. Because this NLLB adapter was **not trained to accept retrieval prompts**, Stage 4 uses conservative output post-editing rather than inserting an unfamiliar prompt into the encoder input. The heuristic can still make incorrect substitutions and should not be considered a production translation system.

The four ablations are NLLB only, NLLB + dictionary, NLLB + TM, and full hybrid. Benchmark scores use the same 160 Kaggle NLLB hypotheses saved in Stage 3, replayed in test-row order and checked against source/reference/type, together with live local train-only retrieval. No test target is used for routing. This is a genuine **offline replay benchmark** of the saved neural outputs and deterministic hybrid logic, not a fresh NLLB inference run. The local `replay_processing_ms` values omit neural generation; the Stage 3 Kaggle T4 mean latency was 273.075916 ms and must not be added to local replay timings as though they were measured on one machine.

```powershell
python scripts/benchmark_hybrid.py --config configs/hybrid.yaml
python scripts/translate_hybrid.py "H’rê text" --variant full_hybrid
```

The second command loads the actual adapter and base model for a new sentence; it needs `[neural]` dependencies and is slow on a CPU-only machine. Results and auditable per-row predictions are saved under `artifacts/evaluation/stage4_results.csv` and `artifacts/evaluation/stage4_predictions/`.

| Model / ablation | Test BLEU | Test chrF++ | TM direct | Medium TM fallback | Dictionary edits |
|---|---:|---:|---:|---:|---:|
| Translation Memory | 4.552510 | 17.809439 | 160 | 0 | 0 |
| NLLB only | 0.435135 | 5.879872 | 0 | 0 | 0 |
| NLLB + dictionary | 0.467314 | 6.888663 | 0 | 0 | 33 |
| NLLB + TM | 1.739116 | 9.176759 | 5 | 12 | 0 |
| Full hybrid | 1.739116 | 9.176226 | 5 | 12 | 1 |

The full hybrid improves on the weak NLLB baseline but does **not** beat the standalone TM baseline. The dictionary edit changes one additional full-hybrid prediction but does not improve corpus chrF++. These are negative findings, not a reason to tune thresholds on the test set.

## Experiment tracking and model management (Stage 5)

MLflow 3 stores the `hre-translate` experiment in local SQLite `mlflow.db`, with artifacts in `mlartifacts/`; both are ignored by Git. Install the tracking extra (shown in Setup), then from the repository root:

```powershell
python scripts/import_historical_runs.py
& "C:\Users\doanh\.conda\envs\hre-translate-stage1\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The explicit Python path works even when Conda's `Scripts` directory is not on PowerShell's `PATH`. After activating the environment, `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` is equivalent. The import is idempotent and does **not** retrain or rerun evaluation: it labels the nine saved Stage 2–4 results as `historical_import`. Their `git_commit` records the import-time checkout, not necessarily the original Kaggle/training commit; `git_commit_scope` makes this explicit. Each run records dataset file SHA-256 hashes and a combined version hash, model type, available hyperparameters/metrics, config, prediction/result artifacts, provenance and latency context. Unavailable metrics are omitted rather than fabricated. The NLLB adapter weights/config are attached to its imported evaluation run.

Future entrypoint executions log new runs by default; use `--no-mlflow` only when tracking is intentionally disabled:

```powershell
python scripts/evaluate.py --model all
python scripts/train_nllb.py --config configs/nllb.yaml
python scripts/evaluate_nllb.py --config configs/nllb.yaml --adapter artifacts/models/nllb/final_adapter
python scripts/benchmark_hybrid.py --config configs/hybrid.yaml
```

The NLLB training command now performs fixed-test evaluation after training so that its run contains BLEU, chrF++, and measured latency. This requires GPU resources and is **not** needed to view the imported Kaggle result. Stage 4 replay runs log local replay-processing latency separately from NLLB generation; they are not eligible for speed-based promotion.

Every run starts as `Candidate`. The current `Champion` tag is on the historical IBM1 run (chrF++ 20.718799, mean latency 0.260700 ms). To review another run, use `python scripts/promote_model.py RUN_ID`; only `--apply` changes tags. A candidate must improve chrF++, use the same dataset version and latency context, and keep mean latency at most the larger of 1.5× Champion latency or Champion latency + 5 ms. Replay/cross-hardware results cannot be promoted by this rule. Tags are an experiment-management concept, **not** a deployed service or registered model artifact.

## Roadmap

1. **Stage 1 — complete:** data inspection, normalization, validation, leakage-aware splitting, and tests.
2. **Stage 2 — complete:** dictionary, longest phrase, Translation Memory, IBM Model 1, and shared evaluation.
3. **Stage 3 — complete:** NLLB-600M LoRA training, fixed-test evaluation, and comparison with classical baselines.
4. **Stage 4 — complete:** train-only retrieval, hybrid routing, replay benchmark, and ablations.
5. **Stage 5 — complete:** MLflow tracking, historical import, and manual Champion/Candidate review.
5. FastAPI and web UI.
6. Docker, CI/CD, Kubernetes, and monitoring.
