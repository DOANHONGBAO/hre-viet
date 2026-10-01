# HRE-TRANSLATE

Low-resource H’rê → Vietnamese machine translation, built as a reproducible research platform and local product demo.

## Overview

HRE-TRANSLATE compares lexical, statistical, retrieval, and neural translation on one fixed held-out test set. The selected default is exposed through FastAPI and a responsive React interface. Training and evaluation artifacts are auditable; MLflow provides local experiment tracking. This is a research demo, not a validated production translator.

## Architecture

```mermaid
flowchart LR
    A[Parallel data] --> B[Normalize and split]
    B --> C[Translation models]
    C --> D[Fixed-test evaluation]
    D --> E[Selected default]
    E --> F[FastAPI]
    F --> G[React web UI]
```

## Translation approaches

- Rule-Based: normalized dictionary lookup, longest phrase match, copied unknowns.
- Statistical MT: IBM Model 1 plus a smoothed Vietnamese bigram language model and small beam decoder.
- Translation Memory: train-only character TF-IDF retrieval.
- Neural / pretrained: NLLB-200 distilled 600M with a LoRA adapter.
- Hybrid retrieval: train-only dictionary/TM routing and conservative neural post-editing.

## Model comparison

All rows use the same 160 held-out test pairs. Scores are corpus BLEU and chrF++ (SacreBLEU); details and the reproducible sample are in [model_comparison.csv](artifacts/evaluation/model_comparison.csv) and [model_outputs.csv](artifacts/evaluation/model_outputs.csv).

| Model | BLEU | chrF++ | Avg / median latency (ms) |
|---|---:|---:|---:|
| Statistical MT + bigram LM | 6.277 | 23.060 | 0.532 / 0.233 |
| IBM Model 1 | 4.249 | 20.719 | 0.272 / 0.141 |
| Translation Memory | 4.553 | 17.809 | 35.036 / 34.643 |
| Rule-Based | 2.331 | 11.724 | 0.047 / 0.032 |
| Dictionary | 2.331 | 11.724 | 0.031 / 0.025 |
| NLLB + TM | 1.739 | 9.177 | Not measured end-to-end |
| Full hybrid | 1.739 | 9.176 | Not measured end-to-end |
| NLLB + dictionary | 0.467 | 6.889 | Not measured end-to-end |
| NLLB + LoRA | 0.435 | 5.880 | 273.076 / 158.469 (Kaggle T4) |
| NLLB only replay | 0.435 | 5.880 | Not measured end-to-end |

Statistical MT is the selected default by chrF++; this is not a claim of general superiority beyond this small test set. NLLB uses verified Kaggle predictions; hybrid replays those predictions with local retrieval. Neural latency was measured on different hardware and is not directly comparable. Rule-Based reuses the dictionary rules, so identical quality scores are expected.

## Tech stack

Python, pandas, scikit-learn, SacreBLEU, PyTorch, Transformers, PEFT, FastAPI, SQLite, MLflow, React, TypeScript, Vite, Tailwind CSS.

## Run locally

Use Python 3.11 or 3.12. From the repository root:

    python -m pip install -e ".[dev,serve]"
    python scripts/benchmark_models.py
    python -m pytest -q --basetemp artifacts/.pytest_tmp/local-run
    python -m uvicorn hre_translate.serving.app:app --host 127.0.0.1 --port 8000

In another terminal:

    cd apps/web
    npm install
    npm run dev

Open http://127.0.0.1:5173. The frontend calls http://127.0.0.1:8000 by default; set VITE_API_BASE_URL for a different backend. API docs are at http://127.0.0.1:8000/docs. Example:

    curl.exe -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" --data-binary '{"text":"Lăm kleq kô pa zâu","model":"default"}'

The auto and default choices use default_model in configs/serving.yaml; explicit rule_based, dictionary, ibm1, statistical, translation_memory, nllb, and hybrid are available when their artifacts/dependencies exist. GET /models reports availability. Feedback is stored locally in SQLite and never automatically added to training. NLLB weights are Git-ignored and require the local adapter plus base model; the other methods run without them.

Install optional neural and tracking dependencies only when needed: `python -m pip install -e ".[neural,tracking]"`. For MLflow:

    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

## Project structure

    apps/web/     React demo
    configs/      Model, benchmark, and serving settings
    data/         Original, processed, and fixed split data
    src/          Translation, evaluation, tracking, and API code
    scripts/      Benchmark and training entrypoints
    tests/        Python tests
    artifacts/    Model and evaluation outputs

## Future work

- Improve H’rê coverage and add independent translation quality review.
- Test new neural initialization strategies on the unchanged split.
- Repeat all latency measurements on one machine before deployment claims.
- Add authentication and deployment infrastructure only after model quality improves.
