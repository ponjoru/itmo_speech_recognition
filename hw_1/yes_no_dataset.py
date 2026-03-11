import torch
import torch.nn as nn
import torchaudio

from pathlib import Path
from typing import Optional, Union
from torchaudio.datasets import SPEECHCOMMANDS


__all__  = ['collate_fn', 'YesNoDataset']


def collate_fn(batch):
    # batch is a list of dicts: {'waveform': Tensor(1,T), 'label': int, 'meta': dict}
    waveforms = [item['waveform'].squeeze(0) for item in batch]  # each (T,)
    labels = [item['label'] for item in batch]
    meta = [item['meta'] for item in batch]

    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)  # (B, T_max)
    labels = torch.tensor(labels)

    return {
        'waveforms': waveforms,
        'labels': labels,
        'meta': meta,
    }
    

class YesNoDataset(SPEECHCOMMANDS):
    """Subset of Speech Commands containing only 'yes' and 'no' labels.

    Args:
        root: Path to the directory where the dataset is found or downloaded.
        subset: One of None, "training", "validation", or "testing".
        transforms: Optional audio transform applied after resampling.
        sample_rate: Target sample rate; audio is resampled if it differs.
    """

    _LABELS = {"yes": 1, "no": 0}

    def __init__(
        self,
        root: Union[str, Path],
        subset: Optional[str] = None,
        transforms: Optional[nn.Module] = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(
            root=root,
            url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=False,
            subset=subset,
        )
        self._walker = [path for path in self._walker if Path(path).parent.name in self._LABELS]
        self.transforms = transforms if transforms is not None else nn.Identity()
        self.sample_rate = sample_rate

    def label2id(self, label: str) -> int:
        return self._LABELS[label]

    def __getitem__(self, n: int):
        waveform, sample_rate, str_label, speaker_id, utterance_number = super().__getitem__(n)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        waveform: torch.Tensor = self.transforms(waveform)
        label: int = self.label2id(str_label)
        meta: dict = {
            'label': str_label,
            'sample_rate': sample_rate,
            'speaker_id': speaker_id,
            'utterance_number': utterance_number,
        }

        return {
            'waveform': waveform,
            'label': label,
            'meta': meta,
        }
