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

## Stage 2 training and evaluation protocol

- Every baseline is trained only on `data/splits/train.parquet` and evaluated on all 160 rows of `data/splits/test.parquet`. Validation and test targets are never indexed by a model.
- Dictionary training uses only `lexicon` and `phrase` rows. Ambiguous source entries are resolved deterministically by frequency, then lexical order.
- Unknown dictionary and IBM1 tokens are copied to the output and reported, making OOV measurable without silently deleting source information.
- Translation Memory uses only `sentence` rows and returns the nearest target as its baseline prediction. Similarity ≥ 0.85 is separately marked as a direct candidate; only 5 of 160 test queries reached that threshold in the current run.

## Why character TF-IDF

- H’rê is low-resource and contains diacritics plus spelling variation. Character n-grams retain subword overlap without requiring a trained tokenizer or a large vocabulary.
- `char_wb` limits n-grams to word boundaries, reducing accidental cross-word matches. The 2–5 range captures short morphemes as well as longer orthographic patterns.
- TF-IDF is deterministic, inexpensive on the current corpus, inspectable, and provides cosine similarity suitable for ranking Translation Memory candidates.

## Statistical baseline

- IBM Model 1 is implemented directly in Python so Stage 2 remains portable on Windows and does not depend on Moses or GIZA++.
- EM runs for 10 fixed iterations. Each learned conditional distribution is normalized and covered by tests.
- Decoding is deliberately word-by-word, exposing the expected weaknesses of Model 1 as a reference baseline.
- A Vietnamese n-gram LM was deferred: 1,275 mixed word/phrase/sentence train pairs are insufficient for a robust general LM, and adding one would require separate smoothing and tuning choices.

## Metrics

- Corpus BLEU and chrF++ use SacreBLEU; chrF++ is configured with word order 2.
- Latency reports per-example mean and p95 wall-clock inference time. It is machine-dependent.
- OOV is reported only for models with an explicit unknown-token concept. It is not applicable to nearest-neighbor retrieval.
