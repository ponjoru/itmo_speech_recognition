import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import heapq
import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from torchaudio.models.decoder import ctc_decoder

# ---------------------------------------------------------------------------
# Beam node types
# ---------------------------------------------------------------------------

@dataclass
class BeamNode:
    """CTC beam hypothesis (no LM)."""
    prefix: tuple           # collapsed token IDs (no blanks)
    last:   Optional[int]   # last emitted non-blank token (None if empty)
    pb:     float           # log P(paths ending in blank    | prefix)
    pnb:    float           # log P(paths ending in non-blank | prefix)

    @property
    def score(self) -> float:
        return _log_add(self.pb, self.pnb)


@dataclass
class BeamNodeLM(BeamNode):
    """CTC beam hypothesis with incremental KenLM state."""
    lm_state:   object          = None
    lm_score:   float           = 0.0
    num_words:  int             = 0
    word_chars: List[str]       = field(default_factory=list)

    def total_score(self, alpha: float, beta: float) -> float:
        return self.score + alpha * self.lm_score + beta * self.num_words


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model_path = lm_model_path
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = torch.argmax(log_probs, dim=-1)    # (T, )
        
        # collapse identical consequetive tokens
        prev = None
        collapsed_token_ids = []
        for t in token_ids:
            t = t.item()
            if t != prev:
                collapsed_token_ids.append(t)
            prev = t
        
        token_ids = [t for t in collapsed_token_ids if t != self.blank_token_id]
        
        # token ids to text
        text = self._ids_to_text(token_ids)
        
        return text

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        T, V = log_probs.shape
        NEG_INF = float('-inf')
        
        def _update(nodes: Dict[tuple, BeamNode], key: tuple, pb_add: float, pnb_add: float) -> None:
            if key in nodes:
                nodes[key].pb  = _log_add(nodes[key].pb,  pb_add)
                nodes[key].pnb = _log_add(nodes[key].pnb, pnb_add)
            else:
                nodes[key] = BeamNode(prefix=key[0], last=key[1], pb=pb_add, pnb=pnb_add)
        
        beam: List[BeamNode] = [BeamNode(prefix=(), last=None, pb=0.0, pnb=NEG_INF)]
        
        for t in range(T):
            lp = log_probs[t]
            candidates: Dict[tuple, BeamNode] = {}
        
            for node in beam:
                # Blank -> same hypothesis, accumulate into pb
                _update(candidates, (node.prefix, node.last), node.score + lp[self.blank_token_id], NEG_INF)
        
                for c in range(V):
                    if c == self.blank_token_id:
                        continue
                    if lp[c] == NEG_INF:
                        continue
                    if c == node.last:
                        # pb paths (preceded by blank) -> new distinct c, extend prefix
                        _update(candidates, (node.prefix + (c,), c), NEG_INF, node.pb + lp[c])
                        # pnb paths (c after c) -> CTC duplicate, same prefix
                        _update(candidates, (node.prefix, c),        NEG_INF, node.pnb + lp[c])
                    else:
                        # any path, different token -> extend prefix
                        _update(candidates, (node.prefix + (c,), c), NEG_INF, node.score + lp[c])
        
            beam = heapq.nlargest(self.beam_width, candidates.values(), key=lambda n: n.score)
        
        result = sorted([(list(n.prefix), n.score) for n in beam], key=lambda x: x[1], reverse=True)
        if return_beams:
            return result
        return self._ids_to_text(result[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model_path:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        T, V = log_probs.shape
        NEG_INF = float('-inf')
        LOG10_TO_NAT = math.log(10)
        word_delim_id = self.processor.tokenizer.get_vocab()[self.word_delimiter]
        init_lm_state = kenlm.State()
        self.lm_model.BeginSentenceWrite(init_lm_state)
        
        def _update(nodes, key, pb_add, pnb_add, lm_state, lm_score, num_words, word_chars):
            if key in nodes:
                nodes[key].pb  = _log_add(nodes[key].pb,  pb_add)
                nodes[key].pnb = _log_add(nodes[key].pnb, pnb_add)
            else:
                nodes[key] = BeamNodeLM(
                    prefix=key[0], 
                    last=key[1], 
                    pb=pb_add, 
                    pnb=pnb_add,
                    lm_state=lm_state, 
                    lm_score=lm_score,
                    num_words=num_words, 
                    word_chars=word_chars
                )
        
        beam = [BeamNodeLM(prefix=(), last=None, pb=0.0, pnb=NEG_INF, lm_state=init_lm_state, lm_score=0.0, num_words=0, word_chars=[])]
        
        for t in range(T):
            lp = log_probs[t]
            candidates = {}
            for node in beam:
                total, prefix, last = node.score, node.prefix, node.last
                _update(candidates, (prefix, last), total + lp[self.blank_token_id], NEG_INF,
                        node.lm_state, node.lm_score, node.num_words, node.word_chars)
                for c in range(V):
                    if c == self.blank_token_id or lp[c] == NEG_INF:
                        continue
                    if c == word_delim_id and node.word_chars:
                        word = ''.join(node.word_chars).lower()
                        next_st = kenlm.State()
                        word_lm = self.lm_model.BaseScore(node.lm_state, word, next_st)
                        new_lm_st, new_lm_sc = next_st, node.lm_score + word_lm * LOG10_TO_NAT
                        new_nwords, new_wchars = node.num_words + 1, []
                    elif c != word_delim_id:
                        new_lm_st, new_lm_sc = node.lm_state, node.lm_score
                        new_nwords, new_wchars = node.num_words, node.word_chars + [self.vocab[c]]
                    else:
                        new_lm_st, new_lm_sc = node.lm_state, node.lm_score
                        new_nwords, new_wchars = node.num_words, []
                    
                    if c == last:
                        _update(candidates, (prefix + (c,), c), NEG_INF, node.pb + lp[c],
                                new_lm_st, new_lm_sc, new_nwords, new_wchars)
                        _update(candidates, (prefix, c), NEG_INF, node.pnb + lp[c],
                                node.lm_state, node.lm_score, node.num_words, node.word_chars)
                    else:
                        _update(candidates, (prefix + (c,), c), NEG_INF, total + lp[c],
                                new_lm_st, new_lm_sc, new_nwords, new_wchars)
            beam = heapq.nlargest(self.beam_width, candidates.values(), key=lambda n: n.total_score(self.alpha, self.beta))
        
        best = max(beam, key=lambda n: n.total_score(self.alpha, self.beta))
        return self._ids_to_text(list(best.prefix))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        rescored = []
        for token_ids, acoustic_score in beams:
            text = self._ids_to_text(token_ids)
            words = text.split()
            if words:
                lm_score = self.lm_model.score(text, bos=True, eos=True) * math.log(10)
                num_words = len(words)
            else:
                lm_score = 0.0
                num_words = 0
            total = acoustic_score + self.alpha * lm_score + self.beta * num_words
            rescored.append((token_ids, total))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return self._ids_to_text(rescored[0][0])

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("assets/hw_2/examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("assets/hw_2/examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("assets/hw_2/examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("assets/hw_2/examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("assets/hw_2/examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("assets/hw_2/examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("assets/hw_2/examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("assets/hw_2/examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path='weights/3-gram.pruned.1e-7.arpa.gz')  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)