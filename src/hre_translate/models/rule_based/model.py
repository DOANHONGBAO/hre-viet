from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

from hre_translate.data.normalize import normalize_hre
from hre_translate.models.dictionary import DictionaryTranslator


class RuleBasedTranslator:
    """Evidence-only rules: normalize, exact/longest lexicon match, copy unknowns."""

    def __init__(self, pairs: Iterable[tuple[str, str]]) -> None:
        self.dictionary = DictionaryTranslator(pairs, preserve_unknown=True)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> RuleBasedTranslator:
        return cls(frame[["hre", "vi"]].itertuples(index=False, name=None))

    def translate(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.dictionary.translate(normalize_hre(text) or "")
        return {
            "translation": result["translation"],
            "model": "rule_based",
            "matched_terms": result["matched_terms"],
            "unknown_terms": result["unknown_terms"],
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
