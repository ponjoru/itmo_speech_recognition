import torch
import kenlm
from pyctcdecode import build_ctcdecoder

from . import vocab as V
from . import text_utils


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
        if lm_path is not None:
            lm = kenlm.Model(lm_path)
            self._decoder = build_ctcdecoder(V.VOCAB, kenlm_model=lm, alpha=alpha, beta=beta)
        else:
            self._decoder = build_ctcdecoder(V.VOCAB)

        self._beam_width = beam_width

    def decode_batch(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        logits_np = log_probs.cpu().numpy()
        results = []
        for i, length in enumerate(lengths.tolist()):
            text = self._decoder.decode(logits_np[i, :length], beam_width=self._beam_width)
            indices = [V.WORD2IDX[w] for w in text.split() if w in V.WORD2IDX]
            results.append(indices)
        return results
