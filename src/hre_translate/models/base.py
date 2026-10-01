from __future__ import annotations

from typing import Any, Protocol, TypedDict


class TranslationResult(TypedDict):
    translation: str
    model: str
    latency_ms: float
    matched_terms: list[dict[str, Any]]
    unknown_terms: list[str]


class BaseTranslator(Protocol):
    def translate(self, text: str) -> dict[str, Any]: ...
