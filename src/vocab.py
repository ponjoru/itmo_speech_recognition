BLANK = "<blank>"

# All unique characters appearing in num2words(n, lang='ru') for n in [1000, 999999]
# Verified empirically: 19 Russian letters + space + blank = 21 tokens total
VOCAB = [
    BLANK,   # index 0 — CTC blank
    " ",     # index 1 — word separator
    "а", "в", "д", "е", "и", "к", "м", "н", "о",
    "п", "р", "с", "т", "ц", "ч", "ш", "ы", "ь", "я",
]

BLANK_IDX = 0
VOCAB_SIZE = len(VOCAB)  # 21

char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for i, c in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    return [char2idx[c] for c in text if c in char2idx]


def decode(indices: list[int], collapse_repeated: bool = True) -> str:
    if collapse_repeated:
        deduped = []
        prev = None
        for idx in indices:
            if idx != prev:
                deduped.append(idx)
            prev = idx
        indices = deduped
    return "".join(idx2char[i] for i in indices if i != BLANK_IDX)
