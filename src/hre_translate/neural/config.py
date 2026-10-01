from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hre_translate.utils.config import project_root


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    name: str
    source_language_proxy: str
    target_language: str


class DataConfig(StrictModel):
    train: str
    validation: str
    test: str


class TrainingConfig(StrictModel):
    output_dir: str
    batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    epochs: float = Field(gt=0)
    learning_rate: float = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_source_length: int = Field(gt=0, le=512)
    max_target_length: int = Field(gt=0, le=512)
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    gradient_checkpointing: bool
    fp16: bool
    seed: int
    logging_steps: int = Field(gt=0)
    save_total_limit: int = Field(gt=0)


class LoraConfig(StrictModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str] = Field(min_length=1)


class GenerationConfig(StrictModel):
    beam_size: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0, le=512)


class EvaluationConfig(StrictModel):
    results: str
    predictions: str
    comparison: str
    classical_results: str


class NLLBConfig(StrictModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoraConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def language_strategy_is_explicit(self) -> NLLBConfig:
        if not self.model.source_language_proxy or not self.model.target_language:
            raise ValueError("source proxy and target language codes are required")
        return self


def load_nllb_config(path: str | Path | None = None) -> tuple[NLLBConfig, Path]:
    import yaml

    root = project_root()
    config_path = Path(path) if path else root / "configs" / "nllb.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = NLLBConfig.model_validate(yaml.safe_load(handle))
    return config, root
