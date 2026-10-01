# Train HRE-TRANSLATE Stage 3 on Kaggle

## Kaggle settings

1. Upload `hre-translate-stage3-kaggle.zip` as a private Kaggle Dataset. Kaggle may expose its contents already extracted.
2. Create a new notebook and attach that Dataset.
3. In **Notebook options**, select a GPU accelerator (T4 is sufficient; P100 is also usable).
4. Enable Internet for the first run so Hugging Face can download `facebook/nllb-200-distilled-600M`. If Internet cannot be enabled, attach a Kaggle Dataset containing the model snapshot and set `model.name` in `configs/nllb.yaml` to that local directory.
5. Keep the notebook private if the training data should not be public.

## Notebook cells

In a Python notebook cell, copy the already extracted Dataset to Kaggle's writable working directory:

```python
from pathlib import Path
import shutil

source = Path("/kaggle/input/datasets/hngboon/hre-translate-stage3")
work = Path("/kaggle/working/hre-translate")
shutil.copytree(source, work, dirs_exist_ok=True)
assert (work / "configs" / "nllb.yaml").exists()
%cd /kaggle/working/hre-translate
```

If your Dataset is still a ZIP, extract it to the same `work` directory first. Install the project in a new Python notebook cell. Kaggle already supplies a CUDA-enabled PyTorch build. The Kaggle image used for the completed run contained incompatible `torchao==0.10.0`; remove that optional package if PEFT reports the version error:

```python
!python -m pip install -q -e ".[neural,tracking]"
!python -m pip uninstall -y torchao
!python scripts/check_neural_setup.py
!nvidia-smi
```

The setup check must report `"device": "cuda"`. Training now evaluates the best adapter on the fixed test split and records a single MLflow run after training:

```python
!python scripts/train_nllb.py --config configs/nllb.yaml
```

Download all reproducibility artifacts from the notebook Output tab after creating one archive:

```python
from pathlib import Path
import zipfile

files = [Path("configs/nllb.yaml"), Path("configs/tracking.yaml")]
files += [p for p in Path("artifacts/models/nllb").rglob("*") if p.is_file()]
files += [p for p in Path("artifacts/evaluation").rglob("*") if p.is_file()]
files += [Path("mlflow.db")]
files += [p for p in Path("mlartifacts").rglob("*") if p.is_file()]
with zipfile.ZipFile("/kaggle/working/hre-nllb-stage3-output.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.as_posix())
```

## Memory fallback

The default is LoRA rank 8, train batch size 2, gradient accumulation 8, and FP16. Gradient checkpointing defaults to false because the Kaggle run failed when its reentrant checkpoint implementation detached the LoRA loss graph. If a Kaggle GPU runs out of memory, change these fields first:

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  max_source_length: 96
  max_target_length: 96
```

This preserves an effective batch of 16 while reducing peak memory. Do not lower the fixed evaluation set or move test rows into training.
