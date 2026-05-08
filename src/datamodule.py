import collections
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as L

from .dataset import ASRDataset


class ASRDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

        self.train_csv = os.path.join(data_root, "train.csv")
        self.dev_csv = os.path.join(data_root, "dev.csv")

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ASRDataset(self.train_csv, self.data_root, augment=self.augment)
        self.val_dataset = ASRDataset(self.dev_csv, self.data_root, augment=False)

    def _weighted_sampler(self) -> WeightedRandomSampler:
        spk_counts = collections.Counter(it["spk_id"] for it in self.train_dataset.items)
        weights = [1.0 / spk_counts[it["spk_id"]] for it in self.train_dataset.items]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        waveforms, targets, spk_ids = zip(*batch)

        wav_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        padded = torch.zeros(len(waveforms), wav_lengths.max().item())
        for i, w in enumerate(waveforms):
            padded[i, : w.shape[0]] = w

        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_flat = torch.cat([torch.tensor(t, dtype=torch.long) for t in targets])

        return padded, wav_lengths, targets_flat, target_lengths, list(spk_ids)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self._weighted_sampler(),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
