# Project progress

## Stage 6 bidirectional local demo (2026-10-02)

- Trained independent IBM1 lexical distributions and target-side smoothed bigram LMs from the unchanged 1,275-row train split for both H’rê → Vietnamese and Vietnamese → H’rê; saved separate directional artifacts without overwriting the earlier root artifact.
- Evaluated the reverse model on all 160 fixed test rows: BLEU 3.178896, chrF++ 25.836528, mean/median latency 0.828972/0.720800 ms, OOV ratio 0.118044. Saved all predictions and a seed-42, non-cherry-picked 24-row sample.
- Added direction to the master comparison table; forward and reverse scores are separated. API selects statistical MT per direction and rejects unsupported reverse model choices. React swaps languages without reload or a model selector.
- Added root local-development scripts and configurable frontend API URL. Python tests: 39 passed; React build and lint passed. No deployment work.

## Stabilization (2026-10-01)

- Polished the React demo into one focused H’rê → Vietnamese translator screen: no model selector, no extra tabs, and a compact optional correction flow. The frontend sends `model: default`; model selection remains available through the API only.
- Added `docs/images/architecture.png` and editable SVG based on the image supplied in `ui/`. The README diagram depicts implemented components and labels deployment/monitoring as future work; the original concept image is retained as a reference.
- Verified desktop/mobile layout in headless Edge, no horizontal mobile overflow, no model selector, a successful live `/translate` request, and frontend build/lint. The backend test suite remains green.

- Inspected the repository, fixed train/validation/test splits, saved NLLB adapter/base, existing metrics, API, and Streamlit prototype. No Seq2Seq-scratch, Transformer-scratch, or ByT5 implementation/checkpoint exists here; these were not benchmarked.
- Added a Rule-Based wrapper reusing the working dictionary/longest-phrase implementation; no unsupported H’rê grammar rules were invented.
- Extended IBM Model 1 with validated save/load and added a small smoothed Vietnamese bigram LM with beam decoding. Saved train-only parameters in `artifacts/models/statistical/`.
- Benchmarked Rule-Based, dictionary, IBM1, statistical+LM, and Translation Memory live on the same 160 test rows. Rechecked NLLB and all four Stage 4 ablation predictions against that identical test split and labeled their latency contexts honestly.
- Wrote `artifacts/evaluation/model_comparison.csv` and a seed-42 reproducible 24-row `model_outputs.csv`; selected statistical MT as the configured default. Scores and limitations are in `DECISIONS.md`.
- Added `auto`/`default` and explicit model choices to FastAPI without duplicating translation algorithms. Replaced the Streamlit prototype as primary UI with responsive React/Vite/Tailwind in `apps/web/`; old `ui/app.py` remains as an archived prototype.
- Python: 37 tests passed. Frontend: TypeScript/Vite build and ESLint passed; local HTTP serving and default translation request succeeded. Browser screenshot QA was unavailable because the desktop UI helper failed. No Kubernetes, monitoring, or new deployment infrastructure was added.

## Completed

- Inspected repository inputs and identified the workbook and audio archive.
- Created the Stage 1 package, configuration, normalization, preprocessing, grouping/splitting, validation, tests, and documentation.
- Preserved original data files and their paths.
- Generated JSON and Markdown inspection reports from the real workbook.
- Generated 194 word, 802 phrase, and 599 sentence pairs after cleaning.
- Generated deterministic group-aware splits: 1,275 train, 160 validation, and 160 test pairs.
- Passed data validation with no exact or severe near-duplicate leakage across splits.
- Passed all 8 pytest tests and Ruff checks.
- Implemented dictionary lookup with case-aware exact matching, longest-phrase matching, and explicit unknown terms.
- Implemented deterministic character TF-IDF Translation Memory with top-k retrieval and a configurable direct-candidate threshold.
- Implemented IBM Model 1 EM training and word-level decoding in pure Python.
- Evaluated all classical baselines on all 160 held-out test pairs with real BLEU, chrF++, latency, and applicable OOV metrics.
- Saved aggregate metrics and auditable per-example predictions under `artifacts/evaluation/`.
- Passed all 14 Stage 1–2 pytest tests and Ruff checks.
- Added validated NLLB configuration with batch size, epochs, learning rate, LoRA rank, gradient accumulation, maximum lengths, beam size, and seed.
- Added CUDA/CPU runtime detection, parallel-dataset formatting, real NLLB tokenizer integration, LoRA training, adapter inference, fixed-test evaluation, and Stage 2 comparison code.
- Smoke-tested the official NLLB tokenizer: `vie_Latn` resolves to token ID 256193 and H’rê/Vietnamese examples produce source IDs and labels.
- Added a reproducible Kaggle training package and GPU notebook commands.
- Passed all 18 Stage 1–3 unit tests and Ruff checks locally.
- Completed a 10-epoch Kaggle training run (400 steps, 1,275 train / 160 validation examples) and imported the best LoRA adapter from step 40.
- Verified all 160 test predictions against the fixed split and recomputed BLEU 0.435135, chrF++ 5.879872, mean latency 273.075916 ms, and p95 latency 537.448112 ms.
- Verified that the final adapter hash equals the best checkpoint hash and saved comparison, training logs, and provenance under `artifacts/evaluation/`.
- Implemented Stage 4 longest-match lexical and char TF-IDF sentence retrieval integration, a structured hybrid service, configurable similarity router, and live-adapter CLI.
- Benchmarked TM, NLLB replay, NLLB + dictionary, NLLB + TM, and full hybrid on the fixed 160-row test set without reference-guided routing. Saved metrics and per-row predictions under `artifacts/evaluation/stage4_*`.
- Full hybrid achieved BLEU 1.739116 / chrF++ 9.176226, versus NLLB-only 0.435135 / 5.879872 and standalone TM 4.552510 / 17.809439. This hybrid is not superior to TM.
- Passed all 23 Stage 1–4 pytest tests and Ruff checks.
- Installed MLflow 3.16.1 and integrated tracking with classical evaluation, NLLB training/evaluation, and the Stage 4 hybrid benchmark.
- Created the `hre-translate` experiment in a local SQLite store and idempotently imported nine saved Stage 2–4 evaluation results with config, dataset hashes, metrics, artifacts, provenance, and latency context.
- Added manual Champion/Candidate review; IBM1 is the initial Champion tag. TM was correctly rejected because its chrF++ is lower. No model was deployed.
- Added Stage 6 FastAPI endpoints for health, models, single/batch translation, and validated feedback stored in local SQLite; API translation delegates to existing dictionary, TM, and hybrid services.
- Added Streamlit demo with source input, output/model/latency, dictionary matches, retrieved examples, and correction/rating feedback.
- Verified local API startup and TM route on a real train sentence; downloaded the NLLB base model and verified one offline CPU inference with the saved LoRA adapter (about 34 seconds cold-start).
- Verified Streamlit starts on localhost and its translation form renders the real API result without exceptions.
- Verified an NLLB request through the running FastAPI server returned HTTP 200 and marked the neural model loaded; all 32 pytest tests and Ruff checks pass.

## In Progress

- None. Stage 6 local serving/demo is complete.

## Next

- Stop after Stage 6. Containers, Kubernetes, and deployment work require a separate request.

## Issues

- The pre-existing `hre-translate` Conda environment is broken/inconsistent. It was not deleted; a clean `hre-translate-stage1` environment is used.
- The word-versus-phrase boundary is an operational whitespace-token rule and may need linguistic review by a H’rê speaker.
- Classical held-out scores are low, which is expected: the dictionary has 77.07% OOV, Translation Memory cannot compose unseen sentences, and IBM1 has no word-order or language model.
- Translation Memory OOV is reported as not applicable rather than zero.
- Local `nvidia-smi` sees a GTX 1060 3 GB, but PyTorch is CPU-only and 3 GB VRAM is insufficient for NLLB-600M training. The completed run used a Kaggle Tesla T4.
- NLLB underperforms all three Stage 2 baselines on the fixed test set. The source-language proxy and very small heterogeneous corpus are likely contributors; the saved output contains generic translations and copied H’rê tokens.
- The returned Kaggle ZIP did not contain the modified runtime config, so exact training settings beyond the saved summary/trainer state cannot be independently verified.
- Stage 4 replays saved Kaggle NLLB predictions for the fixed test set; its local processing latency excludes NLLB generation and is not comparable to the Kaggle T4 latency. Live hybrid inference is available but was not benchmarked end-to-end on this CPU-only machine.
- Historical MLflow runs record the Git checkout at import time; the original Kaggle commit is unavailable. Replay latency is tagged separately and cannot support Champion promotion.
- The NLLB base weights are 2.46 GB and are not committed; fresh clones need a one-time download. CPU inference is slow, and the measured NLLB quality is not suitable for production.
- The Stage 6 app is localhost-only and has no authentication; do not expose it on a public interface. Feedback stays in a Git-ignored SQLite database and is not used for automatic retraining.
