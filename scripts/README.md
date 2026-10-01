# Scripts

Stage 1 commands are exposed as Python modules under `hre_translate.data`.

Stage 2 evaluation is available through `evaluate.py`:

```powershell
python scripts/evaluate.py --model all
```

Stage 3 neural commands:

```powershell
python scripts/check_neural_setup.py
python scripts/train_nllb.py --config configs/nllb.yaml
python scripts/evaluate_nllb.py --config configs/nllb.yaml `
  --adapter artifacts/models/nllb/final_adapter
python scripts/package_kaggle.py
python scripts/import_kaggle_results.py artifacts/kaggle/hre-nllb-stage3-output.zip
```
# Stage 4

`python scripts/benchmark_hybrid.py --config configs/hybrid.yaml` evaluates train-only TM and four NLLB/hybrid ablations on the fixed test set, replaying the source-checked Kaggle NLLB hypotheses. `python scripts/translate_hybrid.py "text" --variant full_hybrid` runs live NLLB + retrieval for new input when neural dependencies and base weights are available. Replay latency excludes NLLB generation.

# Stage 5

`python scripts/import_historical_runs.py` imports saved Stage 2–4 results into the local MLflow experiment without rerunning them. `python scripts/promote_model.py RUN_ID` performs a read-only promotion review; `--apply` changes Champion/Candidate tags only. Evaluation entrypoints log new MLflow runs by default; `--no-mlflow` opts out.

# Stage 6

Run the API with `python -m uvicorn hre_translate.serving.app:app --host 127.0.0.1 --port 8000` and the UI in another terminal with `python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501`. Use the project Python environment; `streamlit.exe` might not be on your PowerShell PATH.
