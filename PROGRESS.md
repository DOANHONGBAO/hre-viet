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

## In Progress

- None. Stage 2 is complete.

## Next

- Stop after Stage 2. Begin NLLB/neural work only on explicit instruction.

## Issues

- The pre-existing `hre-translate` Conda environment is broken/inconsistent. It was not deleted; a clean `hre-translate-stage1` environment is used.
- The word-versus-phrase boundary is an operational whitespace-token rule and may need linguistic review by a H’rê speaker.
- Classical held-out scores are low, which is expected: the dictionary has 77.07% OOV, Translation Memory cannot compose unseen sentences, and IBM1 has no word-order or language model.
- Translation Memory OOV is reported as not applicable rather than zero.
