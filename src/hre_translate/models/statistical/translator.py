from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from hre_translate.data.normalize import normalize_hre, normalize_vietnamese
from hre_translate.models.statistical.ibm1 import IBMModel1
from hre_translate.models.text import detokenize, is_word_token, token_key, tokenize


class BigramLanguageModel:
    """Train-only target-language bigrams with add-alpha smoothing."""

    def __init__(self, *, alpha: float = 0.5) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.counts: dict[str, Counter[str]] = {}
        self.vocabulary: set[str] = set()

    def fit(self, targets: list[str]) -> BigramLanguageModel:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary = {"</s>", "<unk>"}
        for target in targets:
            tokens = [token_key(token) for token in tokenize(target)]
            vocabulary.update(tokens)
            previous = "<s>"
            for current in [*tokens, "</s>"]:
                counts[previous][current] += 1
                previous = current
        self.counts = dict(counts)
        self.vocabulary = vocabulary
        return self

    def log_probability(self, previous: str, current: str) -> float:
        if not self.vocabulary:
            raise RuntimeError("Language model must be fitted")
        current = current if current in self.vocabulary else "<unk>"
        context = self.counts.get(previous, Counter())
        return math.log(
            (context[current] + self.alpha)
            / (sum(context.values()) + self.alpha * len(self.vocabulary))
        )


class StatisticalTranslator:
    """Directional IBM1 candidates rescored by a target-language bigram LM."""

    def __init__(
        self,
        lexical: IBMModel1,
        language_model: BigramLanguageModel,
        *,
        lm_weight: float = 0.15,
        beam_size: int = 4,
        candidates_per_token: int = 3,
        source_lang: str = "hre",
        target_lang: str = "vi",
    ) -> None:
        if lm_weight < 0 or beam_size < 1 or candidates_per_token < 1:
            raise ValueError("Invalid statistical decoder settings")
        self.lexical = lexical
        self.language_model = language_model
        self.lm_weight = lm_weight
        self.beam_size = beam_size
        self.candidates_per_token = candidates_per_token
        if (source_lang, target_lang) not in {("hre", "vi"), ("vi", "hre")}:
            raise ValueError("Statistical translation requires hre↔vi direction")
        self.source_lang = source_lang
        self.target_lang = target_lang

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        iterations: int = 10,
        lm_weight: float = 0.15,
        beam_size: int = 4,
        candidates_per_token: int = 3,
        source_lang: str = "hre",
        target_lang: str = "vi",
    ) -> StatisticalTranslator:
        if (source_lang, target_lang) not in {("hre", "vi"), ("vi", "hre")}:
            raise ValueError("Statistical translation requires hre↔vi direction")
        pairs = frame[[source_lang, target_lang]].astype(str).itertuples(index=False, name=None)
        lexical = IBMModel1(iterations=iterations).fit(pairs)
        lm = BigramLanguageModel().fit(frame[target_lang].astype(str).tolist())
        return cls(
            lexical,
            lm,
            lm_weight=lm_weight,
            beam_size=beam_size,
            candidates_per_token=candidates_per_token,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def translate(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        normalize = normalize_hre if self.source_lang == "hre" else normalize_vietnamese
        source_tokens = tokenize(normalize(text) or "")
        beams: list[tuple[float, list[str], str]] = [(0.0, [], "<s>")]
        unknown_terms: list[str] = []
        for source in source_tokens:
            available = self.lexical.probabilities.get(token_key(source), {})
            candidates = sorted(
                (
                    (target, probability)
                    for target, probability in available.items()
                    if is_word_token(target) == is_word_token(source)
                ),
                key=lambda item: (-item[1], item[0]),
            )[: self.candidates_per_token]
            if not candidates:
                candidates = [(source, 1.0)]
                if is_word_token(source):
                    unknown_terms.append(source)
            expanded = []
            for score, output, previous in beams:
                for target, probability in candidates:
                    target_key = token_key(target)
                    next_score = (
                        score
                        + math.log(max(probability, 1e-12))
                        + self.lm_weight * self.language_model.log_probability(
                            previous, target_key
                        )
                    )
                    expanded.append((next_score, [*output, target], target_key))
            beams = sorted(expanded, key=lambda beam: (-beam[0], beam[1]))[: self.beam_size]
        if beams:
            best = max(
                beams,
                key=lambda beam: beam[0]
                + self.lm_weight * self.language_model.log_probability(beam[2], "</s>"),
            )
            translation = detokenize(best[1])
        else:
            translation = ""
        return {
            "translation": translation,
            "model": "statistical",
            "matched_terms": [],
            "unknown_terms": unknown_terms,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.lexical.save(destination / "ibm1.json")
        metadata = {
            "format": "hre-statistical-v2",
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "alpha": self.language_model.alpha,
            "counts": self.language_model.counts,
            "vocabulary": sorted(self.language_model.vocabulary),
            "lm_weight": self.lm_weight,
            "beam_size": self.beam_size,
            "candidates_per_token": self.candidates_per_token,
        }
        (destination / "decoder.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> StatisticalTranslator:
        source = Path(directory)
        metadata = json.loads((source / "decoder.json").read_text(encoding="utf-8"))
        if metadata.get("format") not in {"hre-statistical-v1", "hre-statistical-v2"}:
            raise ValueError("Unsupported statistical model artifact")
        language_model = BigramLanguageModel(alpha=float(metadata["alpha"]))
        language_model.counts = {
            context: Counter(targets) for context, targets in metadata["counts"].items()
        }
        language_model.vocabulary = set(metadata["vocabulary"])
        return cls(
            IBMModel1.load(source / "ibm1.json"),
            language_model,
            lm_weight=float(metadata["lm_weight"]),
            beam_size=int(metadata["beam_size"]),
            candidates_per_token=int(metadata["candidates_per_token"]),
            source_lang=str(metadata.get("source_lang", "hre")),
            target_lang=str(metadata.get("target_lang", "vi")),
        )
