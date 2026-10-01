# Technical decisions

## Normalization

- Normalize all text to Unicode NFC.
- Trim leading/trailing whitespace, collapse repeated whitespace, and remove spacing before common punctuation or after opening brackets.
- Treat nulls and empty strings as missing.
- Preserve case, diacritics, apostrophes, backticks, and all H’rê-specific characters. No alphabet whitelist is applied.

## Dataset interpretation

- Workbook headers are on Excel row 4; data begins on row 5.
- The `TỪ VỰNG` sheet contains both word and multi-token expression pairs. A pair is classified as a word pair only when both H’rê and Vietnamese sides contain one whitespace token; all other usable entries are phrase pairs. This operational definition is configurable.
- The `CÂU` sheet is the sentence-pair source.
- Numeric sentence IDs are mapped to matching `TU_VUNG/<id>.mp3` ZIP members when present. The archive is not extracted.
- Source filename, sheet, Excel row, record ID, and notes are retained as provenance metadata.

## Deduplication and near-duplicates

- Empty/incomplete pairs and exact normalized `(hre, vi)` duplicates are removed.
- Near-duplicates are detected on the combined bilingual pair using character-boundary TF-IDF n-grams (3–5) and cosine similarity.
- Near-duplicates are not deleted. Connected pairs at similarity ≥ 0.88 receive the same group ID.

## Splitting

- Entire near-duplicate groups are assigned to a split together, preventing direct group leakage.
- A deterministic greedy allocator targets 80% train, 10% validation, and 10% test using seed 42. Exact percentages can vary slightly because groups are indivisible.
- Validation independently rejects cross-split pairs with similarity ≥ 0.96, as well as exact duplicates.

## Environment

- Stage 1 uses Python 3.11 and data/quality dependencies only. No neural-model packages are installed.
- The existing `hre-translate` environment was preserved after it proved inconsistent; work uses the isolated `hre-translate-stage1` environment.

