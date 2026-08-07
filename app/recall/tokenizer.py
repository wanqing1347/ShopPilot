from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English product text without external dictionaries.

    Chinese contiguous runs retain the full phrase and emit character bigrams and
    trigrams. English and numeric terms are emitted as normalized words. This is
    deterministic and suitable for both BM25 and the hashing embedding baseline.
    """

    normalized = normalize_text(text)
    tokens: list[str] = []
    tokens.extend(_WORD_RE.findall(normalized))
    for match in _CJK_RE.finditer(normalized):
        run = match.group(0)
        if len(run) >= 2:
            tokens.append(run)
        for width in (2, 3):
            if len(run) < width:
                continue
            tokens.extend(run[index : index + width] for index in range(len(run) - width + 1))
    return tokens


def unique_tokens(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for text in texts:
        for token in tokenize(text):
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
    return result
