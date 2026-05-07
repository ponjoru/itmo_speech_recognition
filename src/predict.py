"""Generate a Kaggle submission CSV from the test split.

Usage:
    python -m src.predict
        --data_root /path/to/asr-2026-spoken-numbers-recognition-challenge
        --checkpoint checkpoints/best_model.pt
        --config checkpoints/config.json
        --output submission.csv
        [--method beam_lm --lm_path src/lm/numbers_4gram.binary]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import AudioDataset, collate_fn
from .decode import decode_batch
from .model import ConformerCTC
from .text_utils import build_lookup_table, inverse_normalize


def load_model(checkpoint: str, config: str, device: torch.device) -> ConformerCTC:
    with open(config) as f:
        cfg = json.load(f)
    model = ConformerCTC(**cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def run_inference(
    model: ConformerCTC,
    data_root: Path,
    split: str,
    method: str,
    lm_model,
    beam_width: int,
    alpha: float,
    beta: float,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], list[dict]]:
    """Run model inference on a split. Returns (digit_hypotheses, metadata_rows)."""
    dataset = AudioDataset(data_root / f"{split}.csv", data_root, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)
    with open(data_root / f"{split}.csv") as f:
        meta = list(csv.DictReader(f))

    device = next(model.parameters()).device
    hyps: list[str] = []

    with torch.no_grad():
        for features, feat_lengths, _, _ in loader:
            features = features.to(device)
            feat_lengths = feat_lengths.to(device)
            log_probs, out_lengths = model(features, feat_lengths)
            russian_hyps = decode_batch(
                log_probs, out_lengths, method=method,
                lm_model=lm_model, beam_width=beam_width,
                alpha=alpha, beta=beta,
            )
            hyps.extend(inverse_normalize(h) for h in russian_hyps)

    return hyps, meta


def add_inference_args(p: argparse.ArgumentParser) -> None:
    """Register shared inference arguments on an ArgumentParser."""
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=["greedy", "beam", "beam_lm"], default="greedy")
    p.add_argument("--lm_path", default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beam_width", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)


def _load_lm(args: argparse.Namespace):
    if args.method != "beam_lm":
        return None
    import kenlm
    lm = kenlm.Model(args.lm_path)
    print(f"KenLM loaded from {args.lm_path}")
    return lm


def predict(args: argparse.Namespace) -> None:
    """Run inference on the test split and write a Kaggle submission CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

    model = load_model(args.checkpoint, args.config, device)
    print(f"Model loaded  ({model.count_parameters()/1e6:.2f}M params)  device={device}")

    lm_model = _load_lm(args)
    build_lookup_table()

    hyps, meta = run_inference(
        model, data_root, "test", args.method, lm_model,
        args.beam_width, args.alpha, args.beta, args.batch_size, args.num_workers,
    )

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "transcription"])
        for row, hyp in zip(meta, hyps):
            w.writerow([row["filename"], hyp])
    print(f"Submission written → {out}  ({len(hyps)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_inference_args(p)
    p.add_argument("--output", default="submission.csv")
    predict(p.parse_args())
