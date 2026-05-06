"""Greedy and beam search CTC decoders, with optional KenLM language model fusion.

Reuses the beam search logic from hw_2, adapted for word-level KenLM and our vocabulary.
"""
from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .vocab import BLANK_IDX, VOCAB, idx2char, decode as ctc_decode


# ---------------------------------------------------------------------------
# Greedy decoder
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: Tensor) -> str:
    """CTC greedy decode.

    Args:
        log_probs: (T, vocab_size) — log probabilities

    Returns:
        Decoded string (Russian words).
    """
    ids = log_probs.argmax(dim=-1).tolist()
    return ctc_decode(ids, collapse_repeated=True)


# ---------------------------------------------------------------------------
# Beam search (no LM)
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class BeamNode:
    prefix: tuple
    last: Optional[int]
    pb: float    # log P(paths ending in blank | prefix)
    pnb: float   # log P(paths ending in non-blank | prefix)

    @property
    def score(self) -> float:
        return _log_add(self.pb, self.pnb)


def beam_search_decode(log_probs: Tensor, beam_width: int = 50) -> list[tuple[list[int], float]]:
    """CTC beam search (no LM). Returns sorted list of (token_ids, score)."""
    lp = log_probs.cpu().numpy()
    T, V = lp.shape
    NEG_INF = float("-inf")

    def _update(nodes, key, pb_add, pnb_add):
        if key in nodes:
            nodes[key].pb = _log_add(nodes[key].pb, pb_add)
            nodes[key].pnb = _log_add(nodes[key].pnb, pnb_add)
        else:
            nodes[key] = BeamNode(prefix=key[0], last=key[1], pb=pb_add, pnb=pnb_add)

    beam = [BeamNode(prefix=(), last=None, pb=0.0, pnb=NEG_INF)]

    for t in range(T):
        candidates = {}
        for node in beam:
            _update(candidates, (node.prefix, node.last), node.score + lp[t, BLANK_IDX], NEG_INF)
            for c in range(V):
                if c == BLANK_IDX or lp[t, c] == NEG_INF:
                    continue
                if c == node.last:
                    _update(candidates, (node.prefix + (c,), c), NEG_INF, node.pb + lp[t, c])
                    _update(candidates, (node.prefix, c), NEG_INF, node.pnb + lp[t, c])
                else:
                    _update(candidates, (node.prefix + (c,), c), NEG_INF, node.score + lp[t, c])

        beam = heapq.nlargest(beam_width, candidates.values(), key=lambda n: n.score)

    results = sorted(
        [(list(n.prefix), n.score) for n in beam],
        key=lambda x: x[1],
        reverse=True,
    )
    return results


# ---------------------------------------------------------------------------
# Beam search with word-level KenLM shallow fusion
# ---------------------------------------------------------------------------

@dataclass
class BeamNodeLM(BeamNode):
    lm_state: object = None
    lm_score: float = 0.0
    n_words: int = 0
    current_word: list[str] = field(default_factory=list)

    def total_score(self, alpha: float, beta: float) -> float:
        return self.score + alpha * self.lm_score + beta * self.n_words


SPACE_IDX = 1   # index of ' ' in our vocab (see vocab.py)
LOG10_TO_NAT = math.log(10)


def beam_search_with_lm(
    log_probs: Tensor,
    lm_model,
    beam_width: int = 50,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> str:
    """CTC beam search with word-level KenLM shallow fusion.

    Args:
        log_probs:  (T, vocab_size)
        lm_model:   kenlm.Model instance
        beam_width: number of active hypotheses
        alpha:      LM weight
        beta:       word insertion bonus

    Returns:
        Best decoded string (Russian words).
    """
    import kenlm

    lp = log_probs.cpu().numpy()
    T, V = lp.shape
    NEG_INF = float("-inf")

    def _update(nodes, key, pb_add, pnb_add, lm_state, lm_score, n_words, current_word):
        if key in nodes:
            nodes[key].pb = _log_add(nodes[key].pb, pb_add)
            nodes[key].pnb = _log_add(nodes[key].pnb, pnb_add)
        else:
            nodes[key] = BeamNodeLM(
                prefix=key[0], last=key[1], pb=pb_add, pnb=pnb_add,
                lm_state=lm_state, lm_score=lm_score,
                n_words=n_words, current_word=current_word,
            )

    init_state = kenlm.State()
    lm_model.BeginSentenceWrite(init_state)
    beam = [BeamNodeLM(prefix=(), last=None, pb=0.0, pnb=NEG_INF,
                       lm_state=init_state, lm_score=0.0, n_words=0, current_word=[])]

    for t in range(T):
        candidates = {}
        for node in beam:
            _update(candidates, (node.prefix, node.last),
                    node.score + lp[t, BLANK_IDX], NEG_INF,
                    node.lm_state, node.lm_score, node.n_words, node.current_word)

            for c in range(V):
                if c == BLANK_IDX or lp[t, c] == NEG_INF:
                    continue

                char = idx2char[c]

                if c == SPACE_IDX and node.current_word:
                    # Complete the current word and score it with LM
                    word = "".join(node.current_word)
                    next_state = kenlm.State()
                    word_lm = lm_model.BaseScore(node.lm_state, word, next_state)
                    new_lm_st = next_state
                    new_lm_sc = node.lm_score + word_lm * LOG10_TO_NAT
                    new_n_words = node.n_words + 1
                    new_word = []
                else:
                    new_lm_st = node.lm_state
                    new_lm_sc = node.lm_score
                    new_n_words = node.n_words
                    new_word = node.current_word + [char] if c != SPACE_IDX else []

                if c == node.last:
                    _update(candidates, (node.prefix + (c,), c), NEG_INF, node.pb + lp[t, c],
                            new_lm_st, new_lm_sc, new_n_words, new_word)
                    _update(candidates, (node.prefix, c), NEG_INF, node.pnb + lp[t, c],
                            node.lm_state, node.lm_score, node.n_words, node.current_word)
                else:
                    _update(candidates, (node.prefix + (c,), c), NEG_INF, node.score + lp[t, c],
                            new_lm_st, new_lm_sc, new_n_words, new_word)

        beam = heapq.nlargest(
            beam_width, candidates.values(),
            key=lambda n: n.total_score(alpha, beta),
        )

    best = max(beam, key=lambda n: n.total_score(alpha, beta))
    return "".join(idx2char[i] for i in best.prefix if i != BLANK_IDX)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def decode_batch(
    log_probs: Tensor,
    lengths: Tensor,
    method: str = "greedy",
    lm_model=None,
    beam_width: int = 50,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> list[str]:
    """Decode a batch of CTC log-probabilities.

    Args:
        log_probs: (T, B, vocab_size) — CTC convention
        lengths:   (B,) actual sequence lengths
        method:    'greedy' | 'beam' | 'beam_lm'

    Returns:
        List of decoded Russian word strings.
    """
    B = log_probs.shape[1]
    results = []
    for i in range(B):
        t = lengths[i].item()
        lp_i = log_probs[:t, i, :]   # (T_i, vocab_size)
        if method == "greedy":
            results.append(greedy_decode(lp_i))
        elif method == "beam":
            beams = beam_search_decode(lp_i, beam_width=beam_width)
            ids, _ = beams[0]
            results.append(ctc_decode(ids, collapse_repeated=False))
        elif method == "beam_lm":
            assert lm_model is not None, "lm_model required for beam_lm"
            results.append(beam_search_with_lm(lp_i, lm_model, beam_width, alpha, beta))
        else:
            raise ValueError(f"Unknown decode method: {method!r}")
    return results
