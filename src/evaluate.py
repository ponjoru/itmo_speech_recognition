"""Evaluate a trained checkpoint on the dev split and report per-speaker / aggregate CER."""
import argparse
import collections
import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from . import text_utils
from .dataset import TARGET_SR
from .decoder import BeamCTCDecoder, GreedyDecoder
from .metrics import compute_cer, harmonic_mean_cer
from .model import ConformerASR

_IND_SPEAKERS = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}


def _load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0)


@torch.inference_mode()
def run_evaluation(
    model: ConformerASR,
    decoder,
    df: pd.DataFrame,
    data_root: str,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Return a copy of *df* with 'hypothesis' and 'correct' columns added."""
    model.eval()
    hypotheses: list[str] = []

    paths = [os.path.join(data_root, f) for f in df["filename"]]

    for start in tqdm(range(0, len(paths), batch_size), desc="Evaluating"):
        batch_paths = paths[start : start + batch_size]
        waveforms = [_load_audio(p) for p in batch_paths]

        wav_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        padded = torch.zeros(len(waveforms), wav_lengths.max().item())
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        log_probs, out_lengths = model(padded.to(device), wav_lengths.to(device))

        word_indices = decoder.decode_batch(log_probs, out_lengths)
        for idx in word_indices:
            text = text_utils.denormalize(idx)
            hypotheses.append(text if text else "1000")

    result = df.copy()
    result["hypothesis"] = hypotheses
    result["reference"] = result["transcription"].astype(str)
    result["correct"] = result["hypothesis"] == result["reference"]
    return result


def print_report(result: pd.DataFrame) -> None:
    spk_hyps: dict[str, list[str]] = collections.defaultdict(list)
    spk_refs: dict[str, list[str]] = collections.defaultdict(list)

    for _, row in result.iterrows():
        spk_hyps[row["spk_id"]].append(row["hypothesis"])
        spk_refs[row["spk_id"]].append(row["reference"])

    ind_hyps, ind_refs, ood_hyps, ood_refs = [], [], [], []

    col_w = 14
    print(f"\n{'Speaker':{col_w}} {'Samples':>8} {'Accuracy':>9} {'CER':>8}")
    print("-" * (col_w + 30))

    for spk in sorted(spk_hyps):
        h, r = spk_hyps[spk], spk_refs[spk]
        cer = compute_cer(h, r)
        acc = sum(hi == ri for hi, ri in zip(h, r)) / len(r)
        tag = "inD" if spk in _IND_SPEAKERS else "ooD"
        print(f"{spk:{col_w}} {len(r):>8}  {acc:>8.1%}  {cer:>7.4f}  [{tag}]")

        if spk in _IND_SPEAKERS:
            ind_hyps.extend(h); ind_refs.extend(r)
        else:
            ood_hyps.extend(h); ood_refs.extend(r)

    print("-" * (col_w + 30))

    cer_ind = compute_cer(ind_hyps, ind_refs)
    cer_ood = compute_cer(ood_hyps, ood_refs) if ood_hyps else float("nan")
    hmean = harmonic_mean_cer(cer_ind, cer_ood) if ood_hyps else float("nan")

    overall_cer = compute_cer(
        list(result["hypothesis"]), list(result["reference"])
    )
    overall_acc = result["correct"].mean()

    print(f"\n{'Overall':{col_w}} {len(result):>8}  {overall_acc:>8.1%}  {overall_cer:>7.4f}")
    print(f"{'inD CER':{col_w}} {len(ind_refs):>8}  {'':>9}  {cer_ind:>7.4f}")
    if ood_hyps:
        print(f"{'ooD CER':{col_w}} {len(ood_refs):>8}  {'':>9}  {cer_ood:>7.4f}")
        print(f"{'HM-CER':{col_w}} {'':>8}  {'':>9}  {hmean:>7.4f}  ← primary metric")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConformerASR.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    decoder = (
        GreedyDecoder()
        if args.greedy
        else BeamCTCDecoder(
            lm_path=args.lm_path,
            alpha=args.alpha,
            beta=args.beta,
            beam_width=args.beam_width,
        )
    )

    dev_csv = args.dev_csv or os.path.join(args.data_root, "dev.csv")
    df = pd.read_csv(dev_csv)

    result = run_evaluation(model, decoder, df, args.data_root, args.batch_size, device)
    print_report(result)

    if args.output_csv:
        cols = ["filename", "spk_id", "reference", "hypothesis", "correct"]
        result[cols].to_csv(args.output_csv, index=False)
        print(f"\nPer-sample results saved to {args.output_csv}")

    
    
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the dev split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dev_csv", default=None, help="Override path to dev CSV")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_csv", default=None, help="Optional: save per-sample results to CSV")
    # Decoding
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--lm_path", default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beam_width", type=int, default=50)
    args = p.parse_args()

    evaluate(args)

if __name__ == "__main__":
    main()
