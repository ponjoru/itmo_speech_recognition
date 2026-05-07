"""Evaluate a trained model on the dev split.

Usage:
    python -m src.evaluate
        --data_root /path/to/asr-2026-spoken-numbers-recognition-challenge
        --checkpoint checkpoints/best_model.pt
        --config checkpoints/config.json
        [--method beam_lm --lm_path src/lm/numbers_4gram.binary]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .predict import add_inference_args, load_model, run_inference, _load_lm
from .text_utils import build_lookup_table
from .train import compute_cer

IND_SPEAKERS = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate on the dev split: print per-speaker CER and harmonic mean."""
    from prettytable import PrettyTable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

    model = load_model(args.checkpoint, args.config, device)
    print(f"Model loaded  ({model.count_parameters()/1e6:.2f}M params)  device={device}")

    lm_model = _load_lm(args)
    build_lookup_table()

    hyps, meta = run_inference(
        model, data_root, "dev", args.method, lm_model,
        args.beam_width, args.alpha, args.beta, args.batch_size, args.num_workers,
    )

    refs = [r["transcription"] for r in meta]
    spks = [r["spk_id"] for r in meta]

    spk_refs: dict[str, list] = defaultdict(list)
    spk_hyps: dict[str, list] = defaultdict(list)
    for ref, hyp, spk in zip(refs, hyps, spks):
        spk_refs[spk].append(ref)
        spk_hyps[spk].append(hyp)

    per_spk = {
        spk: sum(compute_cer(r, h) for r, h in zip(spk_refs[spk], spk_hyps[spk])) / len(spk_refs[spk])
        for spk in sorted(spk_refs)
    }

    t = PrettyTable(["Speaker", "N", "CER"])
    for spk, cer in per_spk.items():
        t.add_row([spk, len(spk_refs[spk]), f"{cer:.4f}"])
    overall = sum(compute_cer(r, h) for r, h in zip(refs, hyps)) / len(refs)
    t.add_row(["TOTAL", len(refs), f"{overall:.4f}"])
    print(t)

    ind = [v for k, v in per_spk.items() if k in IND_SPEAKERS]
    ood = [v for k, v in per_spk.items() if k not in IND_SPEAKERS]
    if ind and ood:
        ind_avg = sum(ind) / len(ind)
        ood_avg = sum(ood) / len(ood)
        hmean = 2 * ind_avg * ood_avg / (ind_avg + ood_avg)
        print(f"\ninD CER : {ind_avg:.4f}")
        print(f"ooD CER : {ood_avg:.4f}")
        print(f"H-mean  : {hmean:.4f}  ← competition metric")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_inference_args(p)
    evaluate(p.parse_args())
