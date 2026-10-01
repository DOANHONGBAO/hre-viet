from __future__ import annotations

import re
import unicodedata
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%\)\]\}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")
_MISSING_STRINGS = {"", "nan", "none", "null", "<na>"}


def normalize_text(value: Any, *, unicode_form: str = "NFC") -> str | None:
    """Normalize layout without case-folding, accent removal, or alphabet filtering."""
    if value is None:
        return None
    try:
        if value != value:  # NaN, without importing pandas.
            return None
    except (TypeError, ValueError):
        pass

    text = unicodedata.normalize(unicode_form, str(value))
    text = "".join(
        " " if char in "\t\r\n" else char
        for char in text
        if char in "\t\r\n" or unicodedata.category(char) not in {"Cc", "Cs"}
    )
    text = _SPACE_RE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text).strip()
    if text.casefold() in _MISSING_STRINGS:
        return None
    return text or None


def normalize_hre(value: Any) -> str | None:
    return normalize_text(value)


def normalize_vietnamese(value: Any) -> str | None:
    return normalize_text(value)
