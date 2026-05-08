import argparse

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .datamodule import ASRDataModule
from .model import ConformerASR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Conformer ASR on Russian spoken numbers")

    # Data
    p.add_argument("--data_root", required=True, help="Path to dataset root directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_augment", action="store_true", help="Disable audio augmentations")

    # Model
    p.add_argument("--n_mels", type=int, default=80)
    p.add_argument("--conformer_dim", type=int, default=160)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=320)
    p.add_argument("--num_layers", type=int, default=10)
    p.add_argument("--kernel_size", type=int, default=31)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--resume_from", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default=None, help="Override training precision (e.g. '16-mixed')")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    dm = ASRDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
    )

    model = ConformerASR(
        n_mels=args.n_mels,
        conformer_dim=args.conformer_dim,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        depthwise_conv_kernel_size=args.kernel_size,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    precision = args.precision or ("16-mixed" if torch.cuda.is_available() else "32")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=args.grad_clip,
        precision=precision,
        log_every_n_steps=10,
        logger=TensorBoardLogger("logs", name="conformer_asr"),
        callbacks=[
            ModelCheckpoint(
                dirpath=args.ckpt_dir,
                filename="asr-{epoch:03d}-{val/hmean_cer:.4f}",
                monitor="val/hmean_cer",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(monitor="val/hmean_cer", mode="min", patience=10),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(model, dm, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
