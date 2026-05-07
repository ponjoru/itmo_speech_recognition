"""
Label normalization (digits → Russian words) and inverse normalization (Russian words → digits).

The domain is closed: only integers in [1000, 999999] appear as transcriptions.
We use num2words for the forward direction and a full lookup table for the inverse.
"""
from __future__ import annotations

import re
from functools import lru_cache
from num2words import num2words
from tqdm import tqdm


@lru_cache(maxsize=None)
def digit_to_russian(n: int) -> str:
    """Convert an integer to its Russian word form."""
    return num2words(int(n), lang="ru")


def normalize_label(transcription: str | int) -> str:
    """Normalize a digit-string label to Russian words for CTC training."""
    return digit_to_russian(int(transcription))


# ---------------------------------------------------------------------------
# Inverse normalization (Russian words → digit string)
# ---------------------------------------------------------------------------

_LOOKUP: dict[str, str] | None = None


def build_lookup_table() -> dict[str, str]:
    """Build {Russian word form: digit string} for all integers in [1000, 999999].
    Takes ~5 seconds. Result is cached in module-level _LOOKUP.
    """
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = {
            num2words(n, lang="ru"): str(n)
            for n in tqdm(range(1000, 1_000_000), desc="Building lookup table", unit="num", leave=False)
        }
    return _LOOKUP


def inverse_normalize(russian_text: str) -> str:
    """Convert Russian word form back to a digit string.

    1. Exact lookup in the precomputed table (fast path).
    2. Normalize whitespace and retry.
    3. Difflib closest-match fallback for rare decoding errors.
    """
    lookup = build_lookup_table()

    text = russian_text.strip().lower()
    if text in lookup:
        return lookup[text]

    # Normalize whitespace
    text_clean = re.sub(r"\s+", " ", text)
    if text_clean in lookup:
        return lookup[text_clean]

    # Fuzzy fallback — only reached when the model/LM produces an invalid string
    from difflib import get_close_matches
    matches = get_close_matches(text_clean, lookup.keys(), n=1, cutoff=0.5)
    if matches:
        return lookup[matches[0]]

    # Last resort: return the original text (will be counted as wrong in CER)
    return text
