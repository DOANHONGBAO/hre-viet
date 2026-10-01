from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from hre_translate.neural.config import NLLBConfig
from hre_translate.neural.data import load_parallel_frame, parallel_records, tokenize_batch
from hre_translate.neural.metrics import trainer_compute_metrics
from hre_translate.neural.runtime import detect_runtime


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _training_arguments(config: NLLBConfig, output_dir: Path, fp16: bool) -> Any:
    from transformers import Seq2SeqTrainingArguments

    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.training.epochs,
        "per_device_train_batch_size": config.training.batch_size,
        "per_device_eval_batch_size": config.training.eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_ratio": config.training.warmup_ratio,
        "weight_decay": config.training.weight_decay,
        "fp16": fp16,
        "predict_with_generate": True,
        "generation_num_beams": config.generation.beam_size,
        "generation_max_length": config.training.max_target_length,
        "logging_steps": config.training.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "chrf_pp",
        "greater_is_better": True,
        "save_total_limit": config.training.save_total_limit,
        "report_to": "none",
        "seed": config.training.seed,
        "data_seed": config.training.seed,
        "gradient_checkpointing": config.training.gradient_checkpointing,
    }
    parameters = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    values["eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"] = "epoch"
    if "gradient_checkpointing_kwargs" in parameters:
        # Non-reentrant checkpointing supports frozen embeddings with trainable
        # LoRA layers; the reentrant default requires input activations to have
        # requires_grad=True and can detach the loss graph.
        values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return Seq2SeqTrainingArguments(**values)


def train_nllb_lora(config: NLLBConfig, root: Path) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        set_seed,
    )

    runtime = detect_runtime(config.training.fp16)
    set_seed(config.training.seed)
    output_dir = _resolve(root, config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        src_lang=config.model.source_language_proxy,
        tgt_lang=config.model.target_language,
    )
    target_token_id = tokenizer.convert_tokens_to_ids(config.model.target_language)
    if target_token_id is None or target_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Unsupported target language: {config.model.target_language}")

    dtype = torch.float16 if runtime.fp16_enabled else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.name, dtype=dtype)
    model.config.forced_bos_token_id = target_token_id
    model.generation_config.forced_bos_token_id = target_token_id
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
    )
    if config.training.gradient_checkpointing:
        # Register this hook on the unwrapped base model. Registering after PEFT
        # wrapping leaves checkpointed encoder inputs without a gradient path.
        model.enable_input_require_grads()
    model = get_peft_model(model, lora)

    paths = {
        "train": _resolve(root, config.data.train),
        "validation": _resolve(root, config.data.validation),
    }
    frames = {name: load_parallel_frame(path) for name, path in paths.items()}
    datasets = {name: Dataset.from_list(parallel_records(frame)) for name, frame in frames.items()}

    def preprocess(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenize_batch(
            batch,
            tokenizer,
            source_language=config.model.source_language_proxy,
            target_language=config.model.target_language,
            max_source_length=config.training.max_source_length,
            max_target_length=config.training.max_target_length,
        )

    tokenized = {
        name: dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        for name, dataset in datasets.items()
    }
    arguments = _training_arguments(config, output_dir, runtime.fp16_enabled)
    trainer_values: dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        "compute_metrics": trainer_compute_metrics(tokenizer),
    }
    trainer_parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_values[
        "processing_class" if "processing_class" in trainer_parameters else "tokenizer"
    ] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_values)
    train_result = trainer.train()

    final_adapter = output_dir / "final_adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    trainable, total = model.get_nb_trainable_parameters()
    summary = {
        "run_config": config.model_dump(),
        "base_model": config.model.name,
        "source_language_proxy": config.model.source_language_proxy,
        "target_language": config.model.target_language,
        "runtime": runtime.__dict__,
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["validation"]),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percent": round(100.0 * trainable / total, 6),
        "training_metrics": {
            key: float(value) if isinstance(value, (float, int, np.number)) else value
            for key, value in train_result.metrics.items()
        },
        "adapter_path": str(final_adapter),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
