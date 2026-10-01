# Stage 1 progress

## Completed

- Inspected repository inputs and identified the workbook and audio archive.
- Created the Stage 1 package, configuration, normalization, preprocessing, grouping/splitting, validation, tests, and documentation.
- Preserved original data files and their paths.
- Generated JSON and Markdown inspection reports from the real workbook.
- Generated 194 word, 802 phrase, and 599 sentence pairs after cleaning.
- Generated deterministic group-aware splits: 1,275 train, 160 validation, and 160 test pairs.
- Passed data validation with no exact or severe near-duplicate leakage across splits.
- Passed all 8 pytest tests and Ruff checks.

## In Progress

- None. Stage 1 is complete.

## Next

- Stop after Stage 1. Begin classical baseline planning only on explicit instruction.

## Issues

- The pre-existing `hre-translate` Conda environment is broken/inconsistent. It was not deleted; a clean `hre-translate-stage1` environment is used.
- The word-versus-phrase boundary is an operational whitespace-token rule and may need linguistic review by a H’rê speaker.
