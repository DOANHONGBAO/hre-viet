from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from hre_translate.data.normalize import normalize_hre
from hre_translate.models.dictionary import DictionaryTranslator
from hre_translate.models.retrieval import TranslationMemory
from hre_translate.models.text import detokenize, token_key, tokenize


class Translator(Protocol):
    def translate(self, text: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RetrievalResult:
    normalized_query: str
    terms: list[dict[str, Any]]
    unknown_terms: list[str]
    examples: list[dict[str, Any]]
    similarity: float


class HybridTranslator:
    """Reference-free routing and conservative retrieval-guided post-editing.

    NLLB was not trained to accept prompts. Retrieval is therefore applied as a
    deterministic post-edit/fallback, never by prepending context to model input.
    """

    VARIANTS = {"nllb_only", "nllb_dictionary", "nllb_tm", "full_hybrid"}

    def __init__(
        self,
        lexicon: DictionaryTranslator,
        memory: TranslationMemory,
        neural: Translator,
        *,
        medium_threshold: float = 0.50,
        high_threshold: float = 0.85,
        top_k: int = 5,
        variant: str = "full_hybrid",
    ) -> None:
        if not 0 <= medium_threshold < high_threshold <= 1:
            raise ValueError("Require 0 <= medium_threshold < high_threshold <= 1")
        if top_k <= 0 or variant not in self.VARIANTS:
            raise ValueError("Invalid top_k or variant")
        self.lexicon = lexicon
        self.memory = memory
        self.neural = neural
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.top_k = top_k
        self.variant = variant

    def retrieve(self, text: str) -> RetrievalResult:
        normalized = normalize_hre(text) or ""
        use_lexicon = self.variant in {"nllb_dictionary", "full_hybrid"}
        use_memory = self.variant in {"nllb_tm", "full_hybrid"}
        lexical = self.lexicon.translate(normalized) if use_lexicon else {
            "matched_terms": [], "unknown_terms": []
        }
        examples = self.memory.retrieve(normalized, self.top_k) if normalized and use_memory else []
        return RetrievalResult(
            normalized_query=normalized,
            terms=lexical["matched_terms"],
            unknown_terms=lexical["unknown_terms"],
            examples=examples,
            similarity=float(examples[0]["similarity"]) if examples else 0.0,
        )

    @staticmethod
    def _replace_copied_terms(translation: str, terms: list[dict[str, Any]]) -> str:
        """Replace source tokens still copied into output; never append unrelated terms."""
        tokens = tokenize(translation)
        ordered = sorted(
            (
                ([token_key(t) for t in tokenize(str(term["source"]))], str(term["target"]))
                for term in terms
            ),
            key=lambda item: -len(item[0]),
        )
        index = 0
        output: list[str] = []
        while index < len(tokens):
            match = next(
                (
                    item
                    for item in ordered
                    if [token_key(t) for t in tokens[index : index + len(item[0])]] == item[0]
                ),
                None,
            )
            if match is None:
                output.append(tokens[index])
                index += 1
            else:
                output.extend(tokenize(match[1]))
                index += len(match[0])
        return detokenize(output)

    @staticmethod
    def _copied_source(translation: str, source: str) -> bool:
        source_tokens = {token_key(token) for token in tokenize(source) if len(token) >= 3}
        output_tokens = {token_key(token) for token in tokenize(translation)}
        return bool(
            source_tokens and len(source_tokens & output_tokens) / len(source_tokens) >= 0.5
        )

    def translate(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        retrieved = self.retrieve(text)
        uses_tm = self.variant in {"nllb_tm", "full_hybrid"}
        uses_dictionary = self.variant in {"nllb_dictionary", "full_hybrid"}
        high = uses_tm and retrieved.examples and retrieved.similarity >= self.high_threshold
        if high:
            translation = str(retrieved.examples[0]["target"])
            model = "translation_memory"
        else:
            neural_result = self.neural.translate(retrieved.normalized_query)
            translation = str(neural_result["translation"])
            model = "nllb_lora"
            medium = uses_tm and retrieved.similarity >= self.medium_threshold
            if medium and self._copied_source(translation, retrieved.normalized_query):
                translation = str(retrieved.examples[0]["target"])
                model = "retrieval_augmented_tm_fallback"
            elif uses_dictionary and (self.variant != "full_hybrid" or medium):
                edited = self._replace_copied_terms(translation, retrieved.terms)
                if edited != translation:
                    translation = edited
                    model = "retrieval_augmented_dictionary"
        return {
            "translation": translation,
            "model": model,
            "retrieved_terms": retrieved.terms,
            "retrieved_examples": retrieved.examples,
            "similarity": retrieved.similarity,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
