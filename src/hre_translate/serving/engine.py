from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

from hre_translate.hybrid import HybridTranslator
from hre_translate.models.dictionary import DictionaryTranslator
from hre_translate.models.retrieval import TranslationMemory
from hre_translate.models.rule_based import RuleBasedTranslator
from hre_translate.models.statistical import IBMModel1, StatisticalTranslator
from hre_translate.neural.config import NLLBConfig, load_nllb_config
from hre_translate.neural.inference import NLLBTranslator
from hre_translate.neural.runtime import detect_runtime
from hre_translate.serving.schemas import TranslateRequest, TranslateResponse
from hre_translate.utils.config import load_config, project_root, resolve


class ModelUnavailableError(RuntimeError):
    pass


class UnsupportedDirectionError(ValueError):
    pass


class LazyNLLBTranslator:
    """Load local adapter/base once; serialize inference to limit CPU/GPU memory use."""

    def __init__(self, adapter_path: Path, config: NLLBConfig, base_model_path: Path) -> None:
        self.adapter_path = adapter_path
        self.config = config
        self.base_model_path = base_model_path
        self._model: NLLBTranslator | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        adapter_ready = (
            (self.adapter_path / "adapter_model.safetensors").is_file()
            and (self.adapter_path / "adapter_config.json").is_file()
            and all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))
        )
        if not adapter_ready:
            return False
        if (self.base_model_path / "config.json").is_file() and (
            self.base_model_path / "pytorch_model.bin"
        ).is_file():
            return True
        try:
            from huggingface_hub import try_to_load_from_cache

            return all(
                isinstance(try_to_load_from_cache(self.config.model.name, filename), str)
                for filename in ("config.json", "pytorch_model.bin")
            )
        except ImportError:
            return False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def translate(self, text: str) -> dict[str, Any]:
        if not self.available:
            raise ModelUnavailableError("Local NLLB adapter or neural dependencies are missing")
        with self._lock:
            if self._model is None:
                try:
                    model_name = (
                        str(self.base_model_path)
                        if (self.base_model_path / "pytorch_model.bin").is_file()
                        else self.config.model.name
                    )
                    self._model = NLLBTranslator.from_adapter(
                        self.adapter_path,
                        model_name=model_name,
                        source_language=self.config.model.source_language_proxy,
                        target_language=self.config.model.target_language,
                        device=detect_runtime(self.config.training.fp16).device,
                        beam_size=self.config.generation.beam_size,
                        max_source_length=self.config.training.max_source_length,
                        max_new_tokens=self.config.generation.max_new_tokens,
                        local_files_only=True,
                    )
                except (OSError, ImportError, RuntimeError) as exc:
                    raise ModelUnavailableError(
                        "Cannot load NLLB locally; check adapter, base-model cache, "
                        "and neural packages"
                    ) from exc
            return self._model.translate(text)


class ServingEngine:
    """Thin serving adapter around Stage 2 models and the Stage 4 hybrid router."""

    def __init__(
        self,
        lexicon: DictionaryTranslator,
        memory: TranslationMemory,
        neural: Any,
        *,
        medium_threshold: float,
        high_threshold: float,
        top_k: int,
        rule_based: RuleBasedTranslator | None = None,
        ibm1: IBMModel1 | None = None,
        statistical: StatisticalTranslator | None = None,
        reverse_statistical: StatisticalTranslator | None = None,
        default_model: str = "hybrid",
        default_models: dict[str, str] | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.memory = memory
        self.neural = neural
        self.top_k = top_k
        self.rule_based = rule_based
        self.ibm1 = ibm1
        self.statistical = statistical
        self.reverse_statistical = reverse_statistical
        self.default_model = default_model
        self.default_models = default_models or {
            "hre_to_vi": default_model, "vi_to_hre": "statistical"
        }
        if default_model not in {
            "dictionary", "rule_based", "ibm1", "statistical", "translation_memory", "hybrid"
        }:
            raise ValueError(f"Unsupported configured default model: {default_model}")
        self.hybrid = {
            name: HybridTranslator(
                lexicon,
                memory,
                neural,
                medium_threshold=medium_threshold,
                high_threshold=high_threshold,
                top_k=top_k,
                variant=name,
            )
            for name in HybridTranslator.VARIANTS
        }

    @classmethod
    def from_project(
        cls, root: Path | None = None, config_path: str = "configs/serving.yaml"
    ) -> ServingEngine:
        root = root or project_root()
        serving, _ = load_config(resolve(root, config_path))
        hybrid, _ = load_config(resolve(root, serving["hybrid_config"]))
        nllb, _ = load_nllb_config(resolve(root, serving["nllb_config"]))
        train = pd.read_parquet(resolve(root, hybrid["data"]["train"]))
        lexicon = DictionaryTranslator.from_frame(
            train.loc[train.dataset_type.isin(hybrid["lexicon_types"])]
        )
        options = hybrid["retrieval"]
        memory = TranslationMemory(
            ngram_range=tuple(options["ngram_range"]),
            direct_match_threshold=float(options["high_threshold"]),
        ).fit(train.loc[train.dataset_type.isin(hybrid["memory_types"])])
        artifacts = serving.get("statistical_artifacts", {})
        statistical_path = resolve(
            root, artifacts.get("hre_to_vi", serving["statistical_artifact"])
        )
        if not (statistical_path / "decoder.json").is_file():
            statistical_path = resolve(root, serving["statistical_artifact"])
        statistical = (
            StatisticalTranslator.load(statistical_path)
            if (statistical_path / "decoder.json").is_file()
            and (statistical_path / "ibm1.json").is_file()
            else StatisticalTranslator.from_frame(train)
        )
        reverse_path = resolve(
            root, artifacts.get("vi_to_hre", "artifacts/models/statistical/vi_to_hre")
        )
        reverse_statistical = (
            StatisticalTranslator.load(reverse_path)
            if (reverse_path / "decoder.json").is_file()
            and (reverse_path / "ibm1.json").is_file()
            else None
        )
        if reverse_statistical is not None and (
            reverse_statistical.source_lang, reverse_statistical.target_lang
        ) != ("vi", "hre"):
            raise ValueError("Reverse statistical artifact has wrong direction")
        return cls(
            lexicon,
            memory,
            LazyNLLBTranslator(
                resolve(root, serving["adapter"]),
                nllb,
                resolve(root, serving["base_model"]),
            ),
            medium_threshold=float(options["medium_threshold"]),
            high_threshold=float(options["high_threshold"]),
            top_k=int(options["top_k"]),
            rule_based=RuleBasedTranslator.from_frame(
                train.loc[train.dataset_type.isin(hybrid["lexicon_types"])]
            ),
            ibm1=statistical.lexical,
            statistical=statistical,
            reverse_statistical=reverse_statistical,
            default_model=str(serving["default_model"]),
            default_models=serving.get("default_models"),
        )

    def models(self) -> list[dict[str, Any]]:
        neural_ready = bool(getattr(self.neural, "available", True))
        return [
            {"id": "auto", "label": f"Auto · {self.default_model}", "available": True},
            {
                "id": "default", "label": f"Selected default · {self.default_model}",
                "available": True,
            },
            {"id": "rule_based", "label": "Rule-Based", "available": self.rule_based is not None},
            {"id": "dictionary", "label": "Dictionary", "available": True},
            {"id": "ibm1", "label": "IBM Model 1", "available": self.ibm1 is not None},
            {
                "id": "statistical", "label": "Statistical MT + bigram LM",
                "available": self.statistical is not None,
            },
            {"id": "translation_memory", "label": "Translation Memory", "available": True},
            {"id": "nllb", "label": "NLLB + LoRA", "available": neural_ready},
            {"id": "hybrid", "label": "Hybrid router (explicit)", "available": True},
        ]

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        started = time.perf_counter()
        direction = f"{request.source}_to_{request.target}"
        choice = (
            self.default_models[direction]
            if request.model in {"auto", "default"} else request.model
        )
        if direction == "vi_to_hre" and choice not in {"statistical", "ibm1"}:
            raise UnsupportedDirectionError(
                f"{choice} is only available for H’rê → Vietnamese; use statistical for reverse"
            )
        if choice == "rule_based":
            if self.rule_based is None:
                raise ModelUnavailableError("Rule-Based model is unavailable")
            answer = self.rule_based.translate(request.text)
            result = {
                "translation": answer["translation"], "model": "rule_based",
                "retrieved_terms": answer["matched_terms"], "retrieved_examples": [],
                "similarity": None,
            }
        elif choice in {"ibm1", "statistical"}:
            if direction == "vi_to_hre":
                model = (
                    self.reverse_statistical.lexical
                    if choice == "ibm1" and self.reverse_statistical is not None
                    else self.reverse_statistical if choice == "statistical" else None
                )
            else:
                model = self.ibm1 if choice == "ibm1" else self.statistical
            if model is None:
                raise ModelUnavailableError(f"{choice} model is unavailable")
            answer = model.translate(request.text)
            result = {
                "translation": answer["translation"], "model": choice,
                "retrieved_terms": answer.get("matched_terms", []),
                "retrieved_examples": [], "similarity": None,
            }
        elif choice == "dictionary":
            result = self.lexicon.translate(request.text)
            result = {
                "translation": result["translation"],
                "model": "dictionary",
                "retrieved_terms": result["matched_terms"],
                "retrieved_examples": [],
                "similarity": None,
            }
        elif choice == "translation_memory":
            result = self.memory.translate(request.text, top_k=self.top_k)
            candidates = result["retrievals"]
            result = {
                "translation": result["translation"],
                "model": "translation_memory",
                "retrieved_terms": [],
                "retrieved_examples": candidates,
                "similarity": candidates[0]["similarity"] if candidates else 0.0,
            }
        else:
            variant = "nllb_only" if choice == "nllb" else "full_hybrid"
            result = self.hybrid[variant].translate(request.text)
        return TranslateResponse(
            translation=str(result["translation"]),
            source=request.source,
            target=request.target,
            model=str(result["model"]),
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieved_terms=result["retrieved_terms"],
            retrieved_examples=result["retrieved_examples"],
            similarity=result.get("similarity"),
        )
