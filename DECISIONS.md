# Technical decisions

## Stabilization: model selection and serving

- The unchanged Stage 1 split has 1,275 train, 160 validation, and 160 test rows. All new live baselines fit only the train split. Historical NLLB and hybrid predictions are accepted only after source/reference/type and row-count checks against the same 160 test rows; hybrid is an offline replay, not a new end-to-end neural run.
- The Rule-Based baseline deliberately reuses the tested dictionary implementation: NFC normalization, exact/casefold lookup, longest phrase priority, and copied unknowns. No H’rê syntactic rule was added without corpus evidence. Its BLEU/chrF++ equal dictionary by design.
- Statistical MT uses the existing deterministic IBM1 EM model (10 maximum iterations) and adds an add-0.5-smoothed Vietnamese bigram LM trained on train targets. For each source token, a beam of 4 combines the top 3 IBM1 lexical candidates with weight 0.15 on LM log-probability. These settings were predeclared in `configs/stabilization.yaml`, not tuned against the test set. The decoder remains monotonic and can produce ungrammatical output; this is not full phrase-based SMT.
- On the 160 fixed test pairs, statistical+LM achieved BLEU 6.277037 and chrF++ 23.059844 (mean 0.532237 ms, median 0.232550 ms in the latest local run). IBM1 reached chrF++ 20.718799, TM 17.809439, Rule-Based/dictionary 11.724265, full hybrid 9.176226, and NLLB 5.879872. Statistical MT is therefore the **selected default model** by chrF++, with BLEU and local latency also acceptable. This is a small-corpus result, not a general superiority claim or independent human evaluation.
- The NLLB 273.075916 ms mean and 158.468711 ms median come from the historical Kaggle T4 run. Full-hybrid end-to-end latency is unavailable because its neural stage was replayed; no cross-machine or partial timings are presented as a controlled comparison. Parameter counts are left blank in the master table rather than inferred for every model.
- `configs/serving.yaml` selects `statistical`; `auto` and `default` resolve to it. Explicit `hybrid` retains the earlier router for experimentation. The service loads the saved statistical artifact when present, and can refit it from train-only data if absent. Neural remains lazy and optional.
- React/Vite/Tailwind is the primary local UI. It queries `/models` to show only available choices, calls FastAPI for translations/feedback, and does not implement translation logic client-side. CORS is restricted to local Vite origins; this is not authorization for public exposure. Streamlit code is retained as a prototype, not the primary app.

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

## Stage 3: unsupported H’rê language code

- The official NLLB-200 model card lists `vie_Latn` but no H’rê code. The initial experiment therefore uses `vie_Latn` as a configurable source proxy and forces `vie_Latn` as the target BOS token.
- The proxy is selected for tokenizer/script compatibility: H’rê data is written in Latin script and contains diacritics shared with Vietnamese. It is not treated as an official H’rê identifier or a linguistic equivalence claim.
- Adding a new language token was rejected for the first experiment because ordinary LoRA on attention projections does not train the embedding for that new token. With only 1,275 train pairs, jointly learning a new language embedding would add instability and a new experimental variable.
- `eng_Latn` is a reasonable neutral-Latin ablation, while `khm_Khmr` is an Austroasiatic but script-mismatched proxy. These should be tested later under the same fixed split if the initial adapter is viable.
- References: the [official NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M) and [NLLB paper](https://arxiv.org/abs/2207.04672).

## Stage 3: LoRA and runtime

- The base model is `facebook/nllb-200-distilled-600M`. LoRA rank 8, alpha 16, dropout 0.1 targets attention `q_proj` and `v_proj`, following PEFT's explicit module-targeting interface.
- FP16, batch size 2, and gradient accumulation 8 were the packaged Kaggle defaults. The completed run disabled gradient checkpointing after the reentrant checkpoint implementation detached the LoRA loss graph. The repository now defaults to that working setting; the training code also supports non-reentrant checkpointing if re-enabled.
- Checkpoints contain adapters and tokenizer files rather than a duplicate of all frozen base weights. The official PEFT documentation confirms Trainer updates and saves adapter parameters for PEFT models.
- The local GTX 1060 has 3 GB VRAM and the installed PyTorch 2.14.1 build is CPU-only. Local execution is limited to unit/smoke tests; Kaggle T4/P100 is the supported training path.
- The model card licenses the checkpoint under CC-BY-NC-4.0, describes it as research-oriented rather than production-ready, and notes degradation beyond its 512-token training length. This project caps both sides at 128 tokens by default.

## Stage 3: evaluation integrity

- Training uses only the fixed 1,275-row train split; checkpoint selection uses the fixed 160-row validation split; final BLEU, chrF++, and latency use the untouched 160-row test split.
- Evaluation warms the model once, then measures synchronized per-example generation latency. It saves every source, reference, hypothesis, and latency for auditing.
- `model_comparison.csv` combines the real NLLB result with the existing Stage 2 CSV. The imported 160 test predictions were matched row-for-row to the fixed test split, and BLEU/chrF++ and latency statistics were recomputed independently.
- The completed run reached 10 epochs and 400 steps. Validation chrF++ peaked at 4.800153 at step 40; `final_adapter` has the same SHA-256 as checkpoint 40. Test BLEU 0.435135 and chrF++ 5.879872 are below every Stage 2 baseline.
- The source proxy is a likely reason for poor lexical transfer, but the current run cannot isolate that factor from corpus size, data quality, or hyperparameters. Treat this as a negative baseline result rather than evidence that NLLB cannot model H’rê.
- The user-returned ZIP contains checkpoints, summary, and predictions but no copy of the modified Kaggle `nllb.yaml`. Thus the recorded model, data counts, epoch count, rank/alpha, and metrics are verified; all other runtime settings have incomplete provenance.
- Inference latency for Stage 3 was measured on Kaggle T4, whereas Stage 2 latency was measured on a local Windows machine. The numeric latency values are descriptive, not a controlled speed comparison.

## Stage 4: hybrid retrieval and benchmark

- Reuse the Stage 2 dictionary longest-match logic and train-sentence `char_wb` TF-IDF retrieval. Both indexes are fitted on the fixed train split only. The test targets and validation split are excluded from retrieval and routing.
- Route by top cosine similarity: ≥0.85 uses the TM target; 0.50–0.85 is the retrieval-augmented path; below 0.50 uses NLLB unchanged. These are predeclared engineering thresholds, not selected by optimizing test BLEU/chrF++.
- The NLLB adapter was not trained with retrieval-formatted inputs. Therefore medium-similarity augmentation is deterministic output post-editing/fallback: replace copied H’rê dictionary spans with their Vietnamese equivalents, or use the nearest TM target when the NLLB hypothesis copies at least half of source word types of length ≥3. This keeps model input on its original training distribution, though it can still introduce errors.
- The ablations isolate saved NLLB output, dictionary post-editing, TM routing/fallback, and their combination. The Stage 3 NLLB test hypotheses are source/reference/type-checked against the unchanged 160-row test set before reuse. This is an honest offline replay benchmark, not a claim of fresh neural generation.
- Report local replay/retrieval processing latency separately from the saved Kaggle T4 generation latency. A controlled end-to-end speed comparison requires running every system on the same hardware and was not available on the CPU-only local environment.
- The full hybrid's BLEU/chrF++ improve over NLLB alone but remain below standalone TM. No test-based threshold tuning was performed after observing the result.

## Stage 5: tracking and promotion

- MLflow 3.16.1 uses a local SQLite backend (`mlflow.db`) and local artifacts (`mlartifacts/`), both Git-ignored. MLflow 3 places the filesystem tracking backend in maintenance mode, so SQLite avoids the file-store opt-in and supports later registry work if needed. Stage 5 uses run tags, not a registered-model alias, because classical and hybrid services have no single packaged MLflow model flavor.
- New CLI evaluations log MLflow runs by default. Existing Stage 2–4 results are imported once as `historical_import`; import does not imply a new training or inference execution. The import-time Git commit and dirty flag are recorded, with a separate scope tag to avoid claiming the original Kaggle commit is known.
- Dataset identity is SHA-256 over the exact train/test files, plus a combined version hash. Config and available hyperparameters, metrics, results/predictions, and provenance are stored per run. Metrics that do not exist, such as end-to-end NLLB latency for Stage 4 replay, are not fabricated.
- A run starts as `Candidate`. `promote_model.py` only changes MLflow tags when explicitly invoked with `--apply`; it never deploys. It requires a strictly better chrF++, the same dataset hash and latency context, and mean latency ≤ max(1.5 × Champion, Champion + 5 ms). The absolute allowance prevents sub-millisecond baselines from making every otherwise viable model ineligible solely due to a tiny absolute increase.
- The historical IBM1 run is the initial `Champion` because it has the highest chrF++ among directly measured local Stage 2 results and low latency. Replay and Kaggle T4 runs are ineligible for promotion against this local latency context. Any production selection would need a separate controlled benchmark and explicit deployment decision.

## Stage 6: local serving and feedback

- FastAPI is a thin transport layer around the existing dictionary, Translation Memory, and `HybridTranslator`; `auto` selects the full hybrid router. NLLB is loaded lazily and guarded by a lock, so high-similarity TM requests do not spend memory or time loading the base model, and concurrent neural requests do not generate simultaneously on limited hardware.
- The API enforces `hre` → `vi`, 2,000-character input limit, and 16-item batch limit. Input validation returns HTTP 422; a missing local NLLB adapter/base returns HTTP 503 while dictionary/TM remain available.
- The local NLLB base is downloaded into Git-ignored `artifacts/models/nllb/base_model/`, and inference uses `local_files_only=True` to prevent surprise network downloads during a request. Adapter and model weights are not committed. One offline CPU smoke inference succeeded; its cold latency is a deployment limitation, not a new benchmark score.
- SQLite is used for feedback because this repository has no PostgreSQL infrastructure. Per-request parameterized inserts store source, prediction, correction, model, UTC timestamp, and optional 1–5 rating. The database is ignored by Git; feedback is not automatically mixed into train/validation/test data.
- Streamlit calls the local API rather than importing model code. Both services bind to localhost in documented commands. No auth, public networking, Docker, or deployment is added in Stage 6.
- This Streamlit page is a basic functional UI, not a final visual design. Keeping presentation in `ui/` and translation behavior behind the API allows later UI redesigns without changing model routing.
