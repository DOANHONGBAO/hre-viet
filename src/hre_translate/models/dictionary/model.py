from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

from hre_translate.data.normalize import normalize_hre, normalize_vietnamese
from hre_translate.models.text import detokenize, is_word_token, token_key, tokenize


def _preferred_target(counts: Counter[str]) -> str:
    return sorted(counts, key=lambda value: (-counts[value], value.casefold(), value))[0]


class DictionaryTranslator:
    """Exact and longest-phrase H're-to-Vietnamese dictionary baseline."""

    def __init__(self, pairs: Iterable[tuple[str, str]], *, preserve_unknown: bool = True) -> None:
        exact_counts: dict[str, Counter[str]] = defaultdict(Counter)
        folded_counts: dict[str, Counter[str]] = defaultdict(Counter)
        phrase_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

        for source, target in pairs:
            source_text = normalize_hre(source)
            target_text = normalize_vietnamese(target)
            if not source_text or not target_text:
                continue
            exact_counts[source_text][target_text] += 1
            folded_counts[source_text.casefold()][target_text] += 1
            source_tokens = tuple(token_key(token) for token in tokenize(source_text))
            if source_tokens:
                phrase_counts[source_tokens][target_text] += 1

        self.exact = {source: _preferred_target(counts) for source, counts in exact_counts.items()}
        self.folded = {
            source: _preferred_target(counts) for source, counts in folded_counts.items()
        }
        self.phrases = {
            source: _preferred_target(counts) for source, counts in phrase_counts.items()
        }
        self.max_phrase_tokens = max((len(key) for key in self.phrases), default=1)
        self.preserve_unknown = preserve_unknown

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, *, preserve_unknown: bool = True
    ) -> DictionaryTranslator:
        return cls(
            frame[["hre", "vi"]].itertuples(index=False, name=None),
            preserve_unknown=preserve_unknown,
        )

    def translate(self, text: str) -> dict[str, Any]:
        normalized = normalize_hre(text)
        if not normalized:
            return {"translation": "", "matched_terms": [], "unknown_terms": []}

        if normalized in self.exact:
            target = self.exact[normalized]
            return {
                "translation": target,
                "matched_terms": [
                    {
                        "source": normalized,
                        "target": target,
                        "start_token": 0,
                        "end_token": len(tokenize(normalized)),
                        "match_type": "exact",
                        "case_sensitive": True,
                    }
                ],
                "unknown_terms": [],
            }
        if normalized.casefold() in self.folded:
            target = self.folded[normalized.casefold()]
            return {
                "translation": target,
                "matched_terms": [
                    {
                        "source": normalized,
                        "target": target,
                        "start_token": 0,
                        "end_token": len(tokenize(normalized)),
                        "match_type": "exact_casefold",
                        "case_sensitive": False,
                    }
                ],
                "unknown_terms": [],
            }

        tokens = tokenize(normalized)
        keys = [token_key(token) for token in tokens]
        output: list[str] = []
        matched: list[dict[str, Any]] = []
        unknown: list[str] = []
        position = 0
        while position < len(tokens):
            match: tuple[int, str] | None = None
            max_length = min(self.max_phrase_tokens, len(tokens) - position)
            for length in range(max_length, 0, -1):
                key = tuple(keys[position : position + length])
                if key in self.phrases:
                    match = (length, self.phrases[key])
                    break
            if match:
                length, target = match
                source = detokenize(tokens[position : position + length])
                output.append(target)
                matched.append(
                    {
                        "source": source,
                        "target": target,
                        "start_token": position,
                        "end_token": position + length,
                        "match_type": "longest_phrase" if length > 1 else "token",
                        "case_sensitive": False,
                    }
                )
                position += length
                continue

            token = tokens[position]
            if is_word_token(token):
                unknown.append(token)
                if self.preserve_unknown:
                    output.append(token)
            else:
                output.append(token)
            position += 1

        return {
            "translation": detokenize(output),
            "matched_terms": matched,
            "unknown_terms": unknown,
        }
