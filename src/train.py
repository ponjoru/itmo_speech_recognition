"""Training script for ConformerCTC Russian spoken numbers ASR.

Usage:
    python -m src.train \
        --data_root /path/to/asr-2026-spoken-numbers-recognition-challenge \
        --output_dir checkpoints \
        --epochs 150 \
        --batch_size 32
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import AudioDataset, collate_fn
from .decode import decode_batch
from .model import build_model
from .text_utils import normalize_label
from .vocab import BLANK_IDX, VOCAB_SIZE


def compute_cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    if not r:
        return float(len(h))
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r):
        d_new = [i + 1] + [0] * len(h)
        for j, hc in enumerate(h):
            d_new[j + 1] = min(
                d_new[j] + 1,
                d[j + 1] + 1,
                d[j] + (0 if rc == hc else 1),
            )
        d = d_new
    return d[len(h)] / len(r)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    samples_meta: list[dict],
) -> tuple[float, dict[str, float]]:
    model.eval()
    all_refs, all_hyps, all_spks = [], [], []
    idx = 0

    with torch.no_grad():
        for features, feat_lengths, labels, label_lengths in loader:
            features = features.to(device)
            feat_lengths = feat_lengths.to(device)
            log_probs, out_lengths = model(features, feat_lengths)
            hyps = decode_batch(log_probs, out_lengths, method="greedy")

            for i in range(features.shape[0]):
                all_refs.append(normalize_label(samples_meta[idx]["transcription"]))
                all_hyps.append(hyps[i])
                all_spks.append(samples_meta[idx].get("spk_id", "unknown"))
                idx += 1

    spk_refs: dict[str, list] = defaultdict(list)
    spk_hyps: dict[str, list] = defaultdict(list)
    for ref, hyp, spk in zip(all_refs, all_hyps, all_spks):
        spk_refs[spk].append(ref)
        spk_hyps[spk].append(hyp)

    per_spk_cer = {
        spk: sum(compute_cer(r, h) for r, h in zip(spk_refs[spk], spk_hyps[spk])) / len(spk_refs[spk])
        for spk in spk_refs
    }
    mean_cer = sum(compute_cer(r, h) for r, h in zip(all_refs, all_hyps)) / len(all_refs)
    return mean_cer, per_spk_cer


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)

    train_dataset = AudioDataset(
        csv_path=data_root / "train.csv",
        data_root=data_root,
        augment=True,
        noise_dir=args.noise_dir,
    )
    dev_dataset = AudioDataset(
        csv_path=data_root / "dev.csv",
        data_root=data_root,
        augment=False,
    )
    with open(data_root / "dev.csv") as f:
        dev_meta = list(csv.DictReader(f))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = build_model(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
    ).to(device)

    config = {
        "vocab_size": VOCAB_SIZE,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "ff_dim": args.ff_dim,
        "conv_kernel": args.conv_kernel,
        "dropout": 0.0,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )

    best_cer = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for features, feat_lengths, labels, label_lengths in train_loader:
            features = features.to(device)
            feat_lengths = feat_lengths.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            log_probs, out_lengths = model(features, feat_lengths)
            loss = ctc_loss(log_probs, labels, out_lengths, label_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            mean_cer, per_spk = evaluate(model, dev_loader, device, dev_meta)
            spk_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(per_spk.items()))
            print(
                f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                f"dev_CER={mean_cer:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"time={elapsed:.0f}s\n  [{spk_str}]"
            )
            history.append({"epoch": epoch, "loss": avg_loss, "dev_cer": mean_cer})

            if mean_cer < best_cer:
                best_cer = mean_cer
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print(f"  → Saved best model (CER={best_cer:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.0f}s")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch{epoch:03d}.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best dev CER: {best_cer:.4f}  →  {output_dir / 'best_model.pt'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--noise_dir", default=None)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=144)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=576)
    p.add_argument("--conv_kernel", type=int, default=31)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
