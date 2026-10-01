# Project progress

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

## In Progress

- None. Stage 5 is complete.

## Next

- Stop after Stage 5. API/deployment work requires a separate request.

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
