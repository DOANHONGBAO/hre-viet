from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

from hre_translate.models.text import detokenize, is_word_token, token_key, tokenize


class IBMModel1:
    """Small, deterministic IBM Model 1 implementation for a classical baseline."""

    def __init__(self, *, iterations: int = 10, null_token: str = "<NULL>") -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.null_token = null_token
        self.probabilities: dict[str, dict[str, float]] = {}

    @staticmethod
    def _sentence_pair(source: str, target: str) -> tuple[list[str], list[str]]:
        return (
            [token_key(token) for token in tokenize(source)],
            [token_key(token) for token in tokenize(target)],
        )

    def fit(self, pairs: Iterable[tuple[str, str]] | pd.DataFrame) -> IBMModel1:
        if isinstance(pairs, pd.DataFrame):
            pair_iterator = pairs[["hre", "vi"]].itertuples(index=False, name=None)
        else:
            pair_iterator = pairs
        corpus = [self._sentence_pair(source, target) for source, target in pair_iterator]
        corpus = [(source, target) for source, target in corpus if source and target]
        if not corpus:
            raise ValueError("IBM Model 1 requires at least one non-empty sentence pair")

        cooccurring: dict[str, set[str]] = defaultdict(set)
        for source_tokens, target_tokens in corpus:
            for source in set(source_tokens + [self.null_token]):
                cooccurring[source].update(target_tokens)
        probabilities = {
            source: {target: 1.0 / len(targets) for target in sorted(targets)}
            for source, targets in cooccurring.items()
        }

        for _ in range(self.iterations):
            counts: dict[str, Counter[str]] = defaultdict(Counter)
            totals: Counter[str] = Counter()
            for source_tokens, target_tokens in corpus:
                source_with_null = source_tokens + [self.null_token]
                for target in target_tokens:
                    denominator = sum(
                        probabilities[source].get(target, 0.0) for source in source_with_null
                    )
                    if denominator == 0.0:
                        continue
                    for source in source_with_null:
                        contribution = probabilities[source].get(target, 0.0) / denominator
                        counts[source][target] += contribution
                        totals[source] += contribution
            for source, target_counts in counts.items():
                total = totals[source]
                probabilities[source] = {
                    target: count / total for target, count in target_counts.items()
                }

        self.probabilities = probabilities
        return self

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        iterations: int = 10,
        null_token: str = "<NULL>",
    ) -> IBMModel1:
        return cls(iterations=iterations, null_token=null_token).fit(frame)

    def probability_sums(self) -> dict[str, float]:
        return {source: sum(targets.values()) for source, targets in self.probabilities.items()}

    def _best_target(self, source_token: str) -> tuple[str, float] | None:
        candidates = self.probabilities.get(token_key(source_token))
        if not candidates:
            return None
        source_is_word = is_word_token(source_token)
        compatible = [
            (target, probability)
            for target, probability in candidates.items()
            if is_word_token(target) == source_is_word
        ]
        if not compatible:
            return None
        return sorted(compatible, key=lambda item: (-item[1], item[0]))[0]

    def translate(self, text: str) -> dict[str, Any]:
        if not self.probabilities:
            raise RuntimeError("IBMModel1.fit must be called before translate")
        source_tokens = tokenize(text)
        output: list[str] = []
        matched: list[dict[str, Any]] = []
        unknown: list[str] = []
        for position, source in enumerate(source_tokens):
            best = self._best_target(source)
            if best is None:
                output.append(source)
                if is_word_token(source):
                    unknown.append(source)
                continue
            target, probability = best
            output.append(target)
            matched.append(
                {
                    "source": source,
                    "target": target,
                    "probability": probability,
                    "source_token": position,
                }
            )
        return {
            "translation": detokenize(output),
            "matched_terms": matched,
            "unknown_terms": unknown,
        }
