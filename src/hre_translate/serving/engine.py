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
from hre_translate.neural.config import NLLBConfig, load_nllb_config
from hre_translate.neural.inference import NLLBTranslator
from hre_translate.neural.runtime import detect_runtime
from hre_translate.serving.schemas import TranslateRequest, TranslateResponse
from hre_translate.utils.config import load_config, project_root, resolve


class ModelUnavailableError(RuntimeError):
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
    ) -> None:
        self.lexicon = lexicon
        self.memory = memory
        self.neural = neural
        self.top_k = top_k
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
        )

    def models(self) -> list[dict[str, Any]]:
        neural_ready = bool(getattr(self.neural, "available", True))
        return [
            {"id": "auto", "label": "Hybrid router", "available": True},
            {"id": "dictionary", "label": "Dictionary", "available": True},
            {"id": "translation_memory", "label": "Translation Memory", "available": True},
            {"id": "nllb", "label": "NLLB + LoRA", "available": neural_ready},
            {"id": "hybrid", "label": "Hybrid router (explicit)", "available": True},
        ]

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        started = time.perf_counter()
        if request.model == "dictionary":
            result = self.lexicon.translate(request.text)
            result = {
                "translation": result["translation"],
                "model": "dictionary",
                "retrieved_terms": result["matched_terms"],
                "retrieved_examples": [],
                "similarity": None,
            }
        elif request.model == "translation_memory":
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
            variant = "nllb_only" if request.model == "nllb" else "full_hybrid"
            result = self.hybrid[variant].translate(request.text)
        return TranslateResponse(
            translation=str(result["translation"]),
            model=str(result["model"]),
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieved_terms=result["retrieved_terms"],
            retrieved_examples=result["retrieved_examples"],
            similarity=result.get("similarity"),
        )
