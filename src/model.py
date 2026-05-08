import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as L

from . import text_utils
from . import vocab as V
from .decoder import GreedyDecoder
from .encoder import ConformerEncoder
from .metrics import compute_cer, harmonic_mean_cer

# Speakers present in the training split — used to split inD vs ooD CER
_IND_SPEAKERS = {"spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F"}


class ConformerASR(L.LightningModule):
    def __init__(
        self,
        # Feature extraction
        n_mels: int = 80,
        hop_length: int = 160,
        win_length: int = 400,
        sample_rate: int = 16_000,
        freq_mask_param: int = 27,
        time_mask_param: int = 40,
        # Encoder
        conformer_dim: int = 160,
        num_heads: int = 4,
        ffn_dim: int = 320,
        num_layers: int = 10,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        # Optimiser
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # --- feature extraction (registered as submodules → moved to device automatically) ---
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=8_000.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        # SpecAugment applied during training only
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param, iid_masks=True
        )

        self.encoder = ConformerEncoder(
            n_mels=n_mels,
            conformer_dim=conformer_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.ctc_head = nn.Linear(conformer_dim, V.VOCAB_SIZE)

        self.ctc_loss = nn.CTCLoss(blank=V.BLANK_IDX, reduction="mean", zero_infinity=True)
        self.greedy_decoder = GreedyDecoder()

        self._val_outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, waveforms: torch.Tensor, wav_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (log_probs, output_lengths).

        log_probs shape: (B, T', vocab_size)
        """
        features = self._extract_features(waveforms)              # (B, T_frames, n_mels)
        frame_lengths = wav_lengths // self.hparams.hop_length + 1

        encoder_out, out_lengths = self.encoder(features, frame_lengths)
        log_probs = F.log_softmax(self.ctc_head(encoder_out), dim=-1)
        return log_probs, out_lengths

    def _extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B, T_audio)
        x = self.mel_transform(waveforms)       # (B, n_mels, T_frames)
        x = self.amplitude_to_db(x)

        if self.training:
            x = self.freq_mask(x)
            x = self.freq_mask(x)
            x = self.time_mask(x)
            x = self.time_mask(x)

        x = x.transpose(1, 2)                  # (B, T_frames, n_mels)

        # Per-utterance mean–variance normalisation
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        waveforms, wav_lengths, targets, target_lengths, _ = batch
        log_probs, out_lengths = self(waveforms, wav_lengths)

        loss = self.ctc_loss(
            log_probs.permute(1, 0, 2),  # (T', B, C) as required by CTCLoss
            targets,
            out_lengths,
            target_lengths,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        waveforms, wav_lengths, targets, target_lengths, spk_ids = batch
        log_probs, out_lengths = self(waveforms, wav_lengths)

        loss = self.ctc_loss(
            log_probs.permute(1, 0, 2), targets, out_lengths, target_lengths
        )
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        hyp_indices = self.greedy_decoder.decode_batch(log_probs, out_lengths)
        hyp_texts = [text_utils.denormalize(idx) for idx in hyp_indices]
        ref_texts = self._split_targets(targets, target_lengths)

        self._val_outputs.append({"hyps": hyp_texts, "refs": ref_texts, "spk_ids": spk_ids})

    def on_validation_epoch_end(self) -> None:
        spk_hyps: dict[str, list[str]] = collections.defaultdict(list)
        spk_refs: dict[str, list[str]] = collections.defaultdict(list)

        for out in self._val_outputs:
            for h, r, s in zip(out["hyps"], out["refs"], out["spk_ids"]):
                spk_hyps[s].append(h)
                spk_refs[s].append(r)

        ind_hyps, ind_refs, ood_hyps, ood_refs = [], [], [], []

        for spk in sorted(spk_hyps):
            cer = compute_cer(spk_hyps[spk], spk_refs[spk])
            self.log(f"val/cer_{spk}", cer, on_epoch=True)
            if spk in _IND_SPEAKERS:
                ind_hyps.extend(spk_hyps[spk])
                ind_refs.extend(spk_refs[spk])
            else:
                ood_hyps.extend(spk_hyps[spk])
                ood_refs.extend(spk_refs[spk])

        cer_ind = compute_cer(ind_hyps, ind_refs)
        # Fall back to inD CER when no ooD speakers are present in this split
        cer_ood = compute_cer(ood_hyps, ood_refs) if ood_hyps else cer_ind
        hmean = harmonic_mean_cer(cer_ind, cer_ood)

        self.log("val/cer_ind", cer_ind, prog_bar=True)
        self.log("val/cer_ood", cer_ood, prog_bar=True)
        self.log("val/hmean_cer", hmean, prog_bar=True)

        self._val_outputs.clear()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_targets(
        self, targets: torch.Tensor, target_lengths: torch.Tensor
    ) -> list[str]:
        """Split flat CTC target tensor back into per-sample digit strings."""
        texts, offset = [], 0
        for length in target_lengths.tolist():
            indices = targets[offset : offset + length].tolist()
            texts.append(text_utils.denormalize(indices))
            offset += length
        return texts
