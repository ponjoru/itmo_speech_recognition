"""Evaluate a trained model on dev or generate a test submission CSV.

Usage — dev, greedy:
    python -m src.evaluate --data_root /path/to/data --checkpoint checkpoints/best_model.pt --config checkpoints/config.json

Usage — dev, beam+LM:
    python -m src.evaluate ... --method beam_lm --lm_path src/lm/numbers_4gram.binary --alpha 0.5 --beta 1.0

Usage — test submission:
    python -m src.evaluate ... --split test --submission_out submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import AudioDataset, collate_fn
from .decode import decode_batch
from .model import ConformerCTC
from .text_utils import build_lookup_table, inverse_normalize
from .train import compute_cer


def load_model(checkpoint: str, config: str, device: torch.device) -> ConformerCTC:
    with open(config) as f:
        cfg = json.load(f)
    model = ConformerCTC(**cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    is_test = args.split == "test"

    model = load_model(args.checkpoint, args.config, device)
    print(f"Model loaded from {args.checkpoint}")

    lm_model = None
    if args.method == "beam_lm":
        import kenlm
        lm_model = kenlm.Model(args.lm_path)
        print(f"KenLM loaded from {args.lm_path}")

    dataset = AudioDataset(data_root / f"{args.split}.csv", data_root, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    with open(data_root / f"{args.split}.csv") as f:
        meta = list(csv.DictReader(f))

    if not is_test:
        print("Building inverse-normalization lookup table...")
        build_lookup_table()

    all_hyps: list[str] = []
    idx = 0
    with torch.no_grad():
        for features, feat_lengths, _, _ in loader:
            features = features.to(device)
            feat_lengths = feat_lengths.to(device)
            log_probs, out_lengths = model(features, feat_lengths)
            russian_hyps = decode_batch(
                log_probs, out_lengths, method=args.method,
                lm_model=lm_model, beam_width=args.beam_width,
                alpha=args.alpha, beta=args.beta,
            )
            for hyp in russian_hyps:
                all_hyps.append(inverse_normalize(hyp))
                idx += 1

    if not is_test:
        from prettytable import PrettyTable

        all_refs = [r["transcription"] for r in meta]
        all_spks = [r.get("spk_id", "unknown") for r in meta]

        spk_refs: dict[str, list] = defaultdict(list)
        spk_hyps: dict[str, list] = defaultdict(list)
        for ref, hyp, spk in zip(all_refs, all_hyps, all_spks):
            spk_refs[spk].append(ref)
            spk_hyps[spk].append(hyp)

        per_spk = {
            spk: sum(compute_cer(r, h) for r, h in zip(spk_refs[spk], spk_hyps[spk])) / len(spk_refs[spk])
            for spk in sorted(spk_refs)
        }

        t = PrettyTable(["Speaker", "N", "CER"])
        for spk, cer in per_spk.items():
            t.add_row([spk, len(spk_refs[spk]), f"{cer:.4f}"])
        overall = sum(compute_cer(r, h) for r, h in zip(all_refs, all_hyps)) / len(all_refs)
        t.add_row(["TOTAL", len(all_refs), f"{overall:.4f}"])
        print(t)

        ind_speakers = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}
        ind = [v for k, v in per_spk.items() if k in ind_speakers]
        ood = [v for k, v in per_spk.items() if k not in ind_speakers]
        if ind and ood:
            ind_avg = sum(ind) / len(ind)
            ood_avg = sum(ood) / len(ood)
            hmean = 2 * ind_avg * ood_avg / (ind_avg + ood_avg)
            print(f"\ninD CER : {ind_avg:.4f}")
            print(f"ooD CER : {ood_avg:.4f}")
            print(f"H-mean  : {hmean:.4f}  ← competition metric")

    if args.submission_out:
        out = Path(args.submission_out)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "transcription"])
            for row, hyp in zip(meta, all_hyps):
                w.writerow([row["filename"], hyp])
        print(f"Submission → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--method", choices=["greedy", "beam", "beam_lm"], default="greedy")
    p.add_argument("--lm_path", default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beam_width", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--submission_out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
