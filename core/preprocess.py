"""Text cleaning, sentence splitting, input limits and window splitting."""
from __future__ import annotations

import re

MAX_CHARS = 5000
# all-MiniLM / mpnet truncate at 256 / 384 word pieces; ~0.75 words per token keeps us safe.
WINDOW_WORDS = 180

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def validate(text: str) -> tuple[bool, str | None]:
    t = clean(text)
    if not t:
        return False, "Paste a paragraph first — the text box is empty."
    if len(t) > MAX_CHARS:
        return False, f"That's {len(t):,} characters. Please shorten it to {MAX_CHARS:,} or fewer."
    return True, None


def counts(text: str) -> tuple[int, int]:
    t = text or ""
    return len(t), len(t.split())


def split_sentences(text: str) -> list[str]:
    t = clean(text)
    if not t:
        return []
    return [s.strip() for s in _SENT_RE.split(t) if s.strip()]


def windows(text: str, max_words: int = WINDOW_WORDS) -> list[str]:
    """Group whole sentences into windows of at most ~max_words words."""
    sents = split_sentences(text)
    out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out or [clean(text)]
