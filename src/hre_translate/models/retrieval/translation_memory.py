from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from hre_translate.data.normalize import normalize_hre


class TranslationMemory:
    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (2, 5),
        direct_match_threshold: float = 0.85,
    ) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self.direct_match_threshold = direct_match_threshold
        self._frame = pd.DataFrame(columns=["hre", "vi"])
        self._matrix: Any = None

    def fit(self, frame: pd.DataFrame) -> TranslationMemory:
        clean = frame.dropna(subset=["hre", "vi"])[["hre", "vi"]].copy()
        clean["hre"] = clean["hre"].map(normalize_hre)
        clean = clean.dropna(subset=["hre"]).drop_duplicates(["hre", "vi"])
        if clean.empty:
            raise ValueError("Translation memory requires at least one training pair")
        self._frame = clean.reset_index(drop=True)
        self._matrix = self.vectorizer.fit_transform(self._frame["hre"])
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._matrix is None:
            raise RuntimeError("TranslationMemory.fit must be called before retrieve")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized = normalize_hre(query)
        if not normalized:
            return []
        query_vector = self.vectorizer.transform([normalized])
        similarities = linear_kernel(query_vector, self._matrix).ravel()
        indices = sorted(
            range(len(self._frame)),
            key=lambda index: (
                -float(similarities[index]),
                str(self._frame.at[index, "hre"]).casefold(),
                str(self._frame.at[index, "vi"]).casefold(),
                index,
            ),
        )[: min(top_k, len(self._frame))]
        return [
            {
                "source": str(self._frame.at[index, "hre"]),
                "target": str(self._frame.at[index, "vi"]),
                "similarity": round(float(similarities[index]), 8),
                "direct_candidate": bool(similarities[index] >= self.direct_match_threshold),
            }
            for index in indices
        ]

    def translate(self, text: str, top_k: int = 5) -> dict[str, Any]:
        candidates = self.retrieve(text, top_k=top_k)
        return {
            "translation": candidates[0]["target"] if candidates else "",
            "retrievals": candidates,
            "direct_candidate": bool(candidates and candidates[0]["direct_candidate"]),
        }
