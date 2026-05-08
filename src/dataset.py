import csv
import os

import torch
import torchaudio
from torch.utils.data import Dataset

from . import text_utils
from .augmentations import get_training_augmentation

TARGET_SR = 16_000


class ASRDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, augment: bool = False) -> None:
        self.data_root = data_root
        self.items: list[dict] = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.items.append({
                    "path": os.path.join(data_root, row["filename"]),
                    "transcription": int(row["transcription"]),
                    "spk_id": row["spk_id"],
                    "src_sr": int(row["samplerate"]),
                })

        self._augmentation = get_training_augmentation(TARGET_SR) if augment else None
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _resampler(self, src_sr: int) -> torchaudio.transforms.Resample:
        if src_sr not in self._resamplers:
            self._resamplers[src_sr] = torchaudio.transforms.Resample(src_sr, TARGET_SR)
        return self._resamplers[src_sr]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int], str]:
        item = self.items[idx]

        waveform, sr = torchaudio.load(item["path"])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            waveform = self._resampler(sr)(waveform)

        waveform = waveform.squeeze(0)  # (T,)

        if self._augmentation is not None:
            arr = waveform.numpy()
            arr = self._augmentation(arr, sample_rate=TARGET_SR)
            waveform = torch.from_numpy(arr)

        target = text_utils.normalize(item["transcription"])
        return waveform, target, item["spk_id"]
