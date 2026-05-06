"""ConformerCTC model for Russian spoken numbers ASR.

Uses torchaudio.models.Conformer as the encoder backbone, with a lightweight
Conv1d subsampling front-end and a linear CTC head.

Default config (d_model=144, 8 layers, 4 heads): ~4.0M parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchaudio.models import Conformer


class ConformerCTC(nn.Module):
    """
    Args:
        vocab_size:   output tokens including CTC blank at index 0
        d_model:      encoder hidden dimension
        n_layers:     number of Conformer layers
        n_heads:      attention heads (d_model must be divisible by n_heads)
        ff_dim:       feed-forward inner dimension (typically 4 * d_model)
        conv_kernel:  depthwise conv kernel size (must be odd)
        dropout:      dropout probability
        n_mels:       input feature dimension
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 144,
        n_layers: int = 8,
        n_heads: int = 4,
        ff_dim: int = 576,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        n_mels: int = 80,
    ):
        super().__init__()

        # 2× time-subsampling: (B, T, n_mels) → (B, T//2, d_model)
        self.subsampling = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=n_heads,
            ffn_dim=ff_dim,
            num_layers=n_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
        )

        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        features: Tensor,
        feat_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            features:     (B, T, n_mels)
            feat_lengths: (B,) actual frame counts

        Returns:
            log_probs:   (T', B, vocab_size)  — time-first for nn.CTCLoss
            out_lengths: (B,) lengths after subsampling
        """
        x = features.transpose(1, 2)          # (B, n_mels, T)
        x = self.subsampling(x)                # (B, d_model, T//2)
        x = x.transpose(1, 2)                  # (B, T//2, d_model)

        out_lengths = (feat_lengths + 1) // 2  # stride-2 output length

        x, out_lengths = self.conformer(x, out_lengths)  # (B, T//2, d_model)

        log_probs = F.log_softmax(self.ctc_head(x), dim=-1)
        return log_probs.transpose(0, 1), out_lengths  # (T', B, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int, **kwargs) -> ConformerCTC:
    model = ConformerCTC(vocab_size=vocab_size, **kwargs)
    n = model.count_parameters()
    print(f"ConformerCTC: {n:,} parameters ({n/1e6:.2f}M)")
    assert n <= 5_000_000, f"Model exceeds 5M parameter limit: {n:,}"
    return model
