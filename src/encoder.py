import torch
import torch.nn as nn
from torchaudio.models import Conformer


class Conv2DSubsampling(nn.Module):
    """4x temporal downsampling via two strided Conv2D layers.

    Treats the mel spectrogram as a 2D image (time × frequency) and applies
    two Conv2D layers with stride 2 in both dimensions, then projects the
    flattened frequency axis to the Conformer model dimension.
    """

    def __init__(self, n_mels: int, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_dim = self._freq_dim(n_mels)
        self.proj = nn.Linear(32 * freq_dim, output_dim)

    @staticmethod
    def _freq_dim(n: int) -> int:
        # output size of Conv2d(kernel=3, stride=2, padding=1): floor((n-1)/2) + 1
        for _ in range(2):
            n = (n - 1) // 2 + 1
        return n

    @staticmethod
    def _subsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        B, T, F = x.shape
        x = x.unsqueeze(1)                              # (B, 1, T, F)
        x = self.conv(x)                                # (B, 32, T', F')
        _, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()         # (B, T', 32, F')
        x = x.view(B, T2, C * F2)                      # (B, T', 32*F')
        x = self.proj(x)                                # (B, T', output_dim)
        return x, self._subsample_lengths(lengths)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        conformer_dim: int = 160,
        num_heads: int = 4,
        ffn_dim: int = 320,
        num_layers: int = 10,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsampling = Conv2DSubsampling(n_mels, conformer_dim)
        self.dropout = nn.Dropout(dropout)
        self.conformer = Conformer(
            input_dim=conformer_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (B, T, n_mels), lengths: (B,) — number of valid frames per sample
        x, lengths = self.subsampling(features, lengths)
        x = self.dropout(x)
        x, lengths = self.conformer(x, lengths)
        return x, lengths
