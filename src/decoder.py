import logging
import torch
import kenlm
from pyctcdecode import build_ctcdecoder

from . import vocab as V
from . import text_utils

# All real words (no blank), used by the DP splitter below.
_VOCAB_WORDS: list[str] = V.VOCAB[1:]


def _parse_decoded_text(text: str) -> list[int]:
    """Convert a pyctcdecode output string to a list of vocab indices.

    pyctcdecode without a space token concatenates tokens directly
    (e.g. "однатысяча" instead of "одна тысяча").  We first try a
    space-split (fast path, covers any future pyctcdecode behaviour
    change), then fall back to a DP segmentation over the known word
    list.
    """
    if not text:
        return []

    # Fast path: already space-separated and every part is a known word.
    parts = text.split()
    if all(p in V.WORD2IDX for p in parts):
        return [V.WORD2IDX[p] for p in parts]

    # DP: segment the concatenated string into known vocabulary words.
    # path[i] holds the word list that covers text[:i], or None if unreachable.
    n = len(text)
    path: list[list[str] | None] = [None] * (n + 1)
    path[0] = []
    for i in range(n):
        if path[i] is None:
            continue
        for word in _VOCAB_WORDS:
            j = i + len(word)
            if j <= n and path[j] is None and text[i:j] == word:
                path[j] = path[i] + [word]

    words = path[n] or []
    return [V.WORD2IDX[w] for w in words]


class GreedyDecoder:
    """Standard CTC greedy (best-path) decoder."""

    def decode_batch(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        # log_probs: (B, T, C),  lengths: (B,) — valid time steps per sample
        argmax = log_probs.argmax(dim=-1)  # (B, T)
        results = []
        for i, length in enumerate(lengths.tolist()):
            seq = argmax[i, :length].tolist()
            collapsed = []
            prev = None
            for tok in seq:
                if tok != prev:
                    if tok != V.BLANK_IDX:
                        collapsed.append(tok)
                    prev = tok
            results.append(collapsed)
        return results


class BeamCTCDecoder:
    """CTC beam-search decoder backed by pyctcdecode + optional KenLM."""

    def __init__(
        self,
        lm_path: str | None = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        beam_width: int = 50,
    ) -> None:
        # pyctcdecode requires blank to be represented as "" (empty string).
        # Our vocab uses "<blank>" at index 0, so we swap it out here only.
        labels = [""] + V.VOCAB[1:]
        if lm_path is not None:
            self._decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
        else:
            self._decoder = build_ctcdecoder(labels)

        self._beam_width = beam_width

    def decode_batch(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        logits_np = log_probs.cpu().numpy()
        results = []
        for i, length in enumerate(lengths.tolist()):
            text = self._decoder.decode(logits_np[i, :length], beam_width=self._beam_width)
            results.append(_parse_decoded_text(text))
        return results
