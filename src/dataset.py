"""AudioDataset for Russian spoken numbers.

Each sample returns (log_mel_features: Tensor[T, 80], label_ids: Tensor[L]).
The collate_fn packs a batch into padded tensors suitable for CTC training.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch import Tensor
from torch.utils.data import Dataset

from .text_utils import normalize_label
from .vocab import encode

TARGET_SR = 16_000
N_MELS = 80
HOP_MS = 10
WIN_MS = 25
HOP_LEN = int(TARGET_SR * HOP_MS / 1000)   # 160
WIN_LEN = int(TARGET_SR * WIN_MS / 1000)   # 400
N_FFT = 512


def _make_mel_transform(sample_rate: int = TARGET_SR) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        win_length=WIN_LEN,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
        f_min=0.0,
        f_max=8000.0,
        mel_scale="htk",
    )


def audio_to_logmel(waveform: Tensor, src_sr: int) -> Tensor:
    """Resample waveform to TARGET_SR, extract 80-dim log-mel features.

    Args:
        waveform: (channels, samples)
        src_sr: original sample rate

    Returns:
        (T, 80) log-mel spectrogram, clipped to avoid log(0)
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if src_sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, src_sr, TARGET_SR)

    mel_transform = _make_mel_transform(TARGET_SR)
    mel = mel_transform(waveform)          # (1, n_mels, T)
    log_mel = torch.log(mel.squeeze(0).T + 1e-9)  # (T, n_mels)
    return log_mel


class AudioDataset(Dataset):
    """Dataset for a single CSV split (train or dev).

    Args:
        csv_path: path to train.csv or dev.csv
        data_root: directory containing the audio files (parent of train/ or dev/)
        augment: apply augmentation during __getitem__ (train only)
        speed_rates: pool of speed factors for perturbation
        noise_dir: optional directory with noise WAV/MP3 files for mixing
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        augment: bool = False,
        speed_rates: tuple[float, ...] = (0.9, 1.0, 1.1),
        noise_dir: str | Path | None = None,
    ):
        self.data_root = Path(data_root)
        self.augment = augment
        self.speed_rates = speed_rates
        self.noise_files: list[Path] = []

        if noise_dir is not None:
            noise_path = Path(noise_dir)
            self.noise_files = list(noise_path.rglob("*.wav")) + list(noise_path.rglob("*.mp3"))

        self.samples: list[dict] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append(row)

        # SpecAugment transforms — aggressive enough to prevent speaker overfitting
        self.freq_mask = T.FrequencyMasking(freq_mask_param=27)
        self.time_mask = T.TimeMasking(time_mask_param=70, iid_masks=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        row = self.samples[idx]
        audio_path = self.data_root / row["filename"]

        waveform, sr = torchaudio.load(str(audio_path))

        # --- Waveform-level augmentations ---
        if self.augment:
            waveform, sr = self._speed_perturb(waveform, sr)
            if self.noise_files and random.random() < 0.5:
                waveform = self._add_noise(waveform, sr)
            elif random.random() < 0.4:
                # Gaussian noise fallback when no noise files are provided
                waveform = self._add_gaussian_noise(waveform)
            if random.random() < 0.35:
                waveform = self._packet_loss(waveform, sr)

        log_mel = audio_to_logmel(waveform, sr)  # (T, 80)

        # --- Feature-level augmentations (SpecAugment) ---
        if self.augment:
            spec = log_mel.T.unsqueeze(0)  # (1, 80, T) for transforms
            # Always apply: 2 freq masks + 3 time masks
            spec = self.freq_mask(self.freq_mask(spec))
            spec = self.time_mask(self.time_mask(self.time_mask(spec)))
            log_mel = spec.squeeze(0).T  # (T, 80)

        # Labels are absent in the test split
        if "transcription" in row:
            label_ids = torch.tensor(encode(normalize_label(row["transcription"])), dtype=torch.long)
        else:
            label_ids = torch.zeros(0, dtype=torch.long)

        return log_mel, label_ids

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _speed_perturb(self, waveform: Tensor, sr: int) -> tuple[Tensor, int]:
        rate = random.choice(self.speed_rates)
        if rate == 1.0:
            return waveform, sr
        # Resample as if audio was recorded at sr*rate, playing back at sr
        # Results in shorter waveform (rate>1 = faster) or longer (rate<1 = slower)
        orig_freq = int(sr * rate)
        waveform = torchaudio.functional.resample(waveform, orig_freq=orig_freq, new_freq=sr)
        return waveform, sr

    def _add_noise(self, waveform: Tensor, sr: int) -> Tensor:
        noise_path = random.choice(self.noise_files)
        try:
            noise, noise_sr = torchaudio.load(str(noise_path))
        except Exception:
            return waveform
        if noise_sr != sr:
            noise = torchaudio.functional.resample(noise, noise_sr, sr)
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        # Trim or tile noise to match waveform length
        n_samples = waveform.shape[1]
        if noise.shape[1] < n_samples:
            repeats = (n_samples // noise.shape[1]) + 1
            noise = noise.repeat(1, repeats)
        noise = noise[:, :n_samples]
        # Random SNR between 5 and 30 dB
        snr_db = random.uniform(5, 30)
        snr = 10 ** (snr_db / 20)
        sig_power = waveform.norm()
        noise_power = noise.norm()
        if noise_power > 0:
            noise = noise * (sig_power / (noise_power * snr))
        return waveform + noise

    def _add_gaussian_noise(self, waveform: Tensor) -> Tensor:
        snr_db = random.uniform(5, 30)
        snr = 10 ** (snr_db / 20)
        sig_power = waveform.norm()
        noise = torch.randn_like(waveform)
        noise = noise * (sig_power / (noise.norm() * snr + 1e-9))
        return waveform + noise

    def _packet_loss(self, waveform: Tensor, sr: int) -> Tensor:
        """Zero out 1–3 random chunks of 50–200ms."""
        waveform = waveform.clone()
        n_samples = waveform.shape[1]
        n_drops = random.randint(1, 3)
        for _ in range(n_drops):
            drop_len = random.randint(int(0.05 * sr), int(0.2 * sr))
            if n_samples <= drop_len:
                continue
            start = random.randint(0, n_samples - drop_len)
            waveform[:, start:start + drop_len] = 0.0
        return waveform


def collate_fn(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad a batch of (log_mel, label_ids) pairs.

    Returns:
        features:       (B, T_max, 80)
        feat_lengths:   (B,) — actual T for each sample (before padding)
        labels:         (B, L_max) — padded with 0
        label_lengths:  (B,) — actual L for each sample
    """
    features, labels = zip(*batch)

    feat_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    T_max = feat_lengths.max().item()
    L_max = label_lengths.max().item()
    B = len(features)

    feat_pad = torch.zeros(B, T_max, N_MELS)
    for i, f in enumerate(features):
        feat_pad[i, : f.shape[0]] = f

    label_pad = torch.zeros(B, max(L_max, 1), dtype=torch.long)
    for i, l in enumerate(labels):
        if l.shape[0] > 0:
            label_pad[i, : l.shape[0]] = l

    return feat_pad, feat_lengths, label_pad, label_lengths
