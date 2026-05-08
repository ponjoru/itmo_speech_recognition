from num2words import num2words
from . import vocab as V

_INVERSE_MAP: dict[str, int] | None = None


def get_inverse_map() -> dict[str, int]:
    """Lazily build a word-sequence → integer lookup for all numbers 1000–999999."""
    global _INVERSE_MAP
    if _INVERSE_MAP is None:
        _INVERSE_MAP = {num2words(n, lang="ru"): n for n in range(1000, 1_000_000)}
    return _INVERSE_MAP


def normalize(number: int) -> list[int]:
    """Convert an integer label to a list of word-vocabulary indices."""
    text = num2words(number, lang="ru")
    return [V.WORD2IDX[w] for w in text.split() if w in V.WORD2IDX]


def denormalize(word_indices: list[int]) -> str:
    """Convert a list of word-vocabulary indices to a digit string.

    Returns an empty string if the decoded word sequence doesn't map to any
    number (e.g. during early training when the model outputs garbage).
    """
    words = [V.IDX2WORD[i] for i in word_indices if i in V.IDX2WORD and i != V.BLANK_IDX]
    text = " ".join(words)
    n = get_inverse_map().get(text)
    return str(n) if n is not None else ""
