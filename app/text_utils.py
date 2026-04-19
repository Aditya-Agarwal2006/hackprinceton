"""Small local text helpers for the app.

These helpers intentionally avoid any hosted model dependency. They are used to
keep the Streamlit demo self-contained now that Gemini has been removed from
the runtime path.
"""

from __future__ import annotations

import re


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_claims_local(text: str) -> list[str]:
    """Split text into short claim-like sentences using simple punctuation.

    This is not a semantic extractor. It is just a lightweight local fallback so
    users can paste AI-generated text and inspect sentence-level claims without
    calling another model.
    """

    if not text or not text.strip():
        return []

    normalized = " ".join(text.strip().split())
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    return parts if parts else [normalized]
