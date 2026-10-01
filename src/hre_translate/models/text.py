from __future__ import annotations

import re
import unicodedata

from hre_translate.data.normalize import normalize_text

_TOKEN_RE = re.compile(r"\w+(?:[-‐‑’']\w+)*|[^\w\s]", flags=re.UNICODE)
_NO_SPACE_BEFORE = set(",.;:!?%)]}")
_NO_SPACE_AFTER = set("([{“‘")


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return _TOKEN_RE.findall(normalized) if normalized else []


def token_key(token: str) -> str:
    return unicodedata.normalize("NFC", token).casefold()


def is_word_token(token: str) -> bool:
    return any(unicodedata.category(char)[0] in {"L", "N"} for char in token)


def detokenize(parts: list[str]) -> str:
    output = ""
    previous = ""
    for part in parts:
        if not output:
            output = part
        elif part and part[0] in _NO_SPACE_BEFORE:
            output += part
        elif previous and previous[-1] in _NO_SPACE_AFTER:
            output += part
        else:
            output += " " + part
        previous = part
    return output.strip()
