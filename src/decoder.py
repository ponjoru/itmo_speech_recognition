import collections

import numpy as np
import torch

from . import vocab as V

_NEG_INF = float("-inf")


def _lse(a: float, b: float) -> float:
    """Numerically stable log-sum-exp of two scalars."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


class GreedyDecoder:
    """Standard CTC greedy (best-path) decoder."""

    def decode_batch(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        # log_probs: (B, T, C),  lengths: (B,)
        argmax = log_probs.argmax(dim=-1)  # (B, T)
        results = []
        for i, length in enumerate(lengths.tolist()):
            seq = argmax[i, :length].tolist()
            collapsed, prev = [], None
            for tok in seq:
                if tok != prev:
                    if tok != V.BLANK_IDX:
                        collapsed.append(tok)
                    prev = tok
            results.append(collapsed)
        return results


class BeamCTCDecoder:
    """CTC prefix beam search (Graves 2012) with optional KenLM shallow fusion.

    Each beam state is a decoded prefix (tuple of word indices) paired with
    two log-probabilities:
      - log_p_b  : total log-prob of paths ending in blank  that produce the prefix
      - log_p_nb : total log-prob of paths NOT ending in blank

    LM integration uses shallow fusion: at each token extension the incremental
    KenLM log-prob (nats) is scaled by *alpha* and a word insertion bonus *beta*
    is added.  Both are zero when no LM path is given.
    """

    def __init__(
        self,
        lm_path: str | None = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        beam_width: int = 50,
    ) -> None:
        self._beam_width = beam_width
        self._alpha = alpha
        self._beta = beta
        self._lm = None
        if lm_path is not None:
            import kenlm
            self._lm = kenlm.Model(lm_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_batch(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        lp_np = log_probs.cpu().numpy()
        return [self._decode_one(lp_np[i, : lengths[i].item()]) for i in range(len(lengths))]

    # ------------------------------------------------------------------
    # Core beam search
    # ------------------------------------------------------------------

    def _decode_one(self, log_probs: np.ndarray) -> list[int]:
        """Beam search over a single (T, C) log-probability array."""
        T, C = log_probs.shape

        # prefix → [log_p_blank, log_p_non_blank]
        beams: dict[tuple, list[float]] = {(): [0.0, _NEG_INF]}
        # Pre-seed the LM score cache with the empty-prefix score of 0.
        lm_cache: dict[tuple, float] = {(): 0.0}

        for t in range(T):
            lp = log_probs[t]  # (C,)
            new_beams: dict[tuple, list[float]] = collections.defaultdict(
                lambda: [_NEG_INF, _NEG_INF]
            )

            for prefix, (pb, pnb) in beams.items():
                p_total = _lse(pb, pnb)
                last = prefix[-1] if prefix else None

                # ── blank: prefix unchanged ──────────────────────────────────
                slot = new_beams[prefix]
                slot[0] = _lse(slot[0], p_total + lp[V.BLANK_IDX])

                # ── non-blank token c ────────────────────────────────────────
                for c in range(1, C):
                    if c != last:
                        # Distinct token → append to prefix
                        ext = prefix + (c,)
                        slot2 = new_beams[ext]
                        slot2[1] = _lse(
                            slot2[1],
                            p_total + lp[c] + self._lm_bonus(prefix, c, lm_cache),
                        )
                    else:
                        # Repeated last token: two sub-cases
                        ext = prefix + (c,)
                        # (a) extend — prior path must have ended in blank
                        slot2 = new_beams[ext]
                        slot2[1] = _lse(
                            slot2[1],
                            pb + lp[c] + self._lm_bonus(prefix, c, lm_cache),
                        )
                        # (b) stay — collapse repeated non-blank (no new word)
                        slot[1] = _lse(slot[1], pnb + lp[c])

            # Keep top-k by total log-prob
            beams = dict(
                sorted(
                    new_beams.items(),
                    key=lambda kv: _lse(kv[1][0], kv[1][1]),
                    reverse=True,
                )[: self._beam_width]
            )

        if not beams:
            return []
        best = max(beams.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]))
        return list(best[0])

    # ------------------------------------------------------------------
    # LM scoring
    # ------------------------------------------------------------------

    def _lm_bonus(self, prefix: tuple, c: int, cache: dict[tuple, float]) -> float:
        """Alpha-scaled incremental KenLM log-prob (nats) + word insertion bonus."""
        if self._lm is None:
            return 0.0

        new_prefix = prefix + (c,)
        if new_prefix not in cache:
            words = " ".join(V.IDX2WORD[i] for i in new_prefix)
            # kenlm returns log10 probability; convert to nats
            cache[new_prefix] = self._lm.score(words, bos=True, eos=False) * np.log(10)

        # cache[prefix] is always populated: either pre-seeded (empty prefix)
        # or set when `prefix` itself was first created as a new_prefix.
        delta = cache[new_prefix] - cache[prefix]
        return self._alpha * delta + self._beta
