import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, conv_groups=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, groups=conv_groups)
        self.conv2 = nn.Conv1d(64, out_channels, kernel_size=5, padding=2, groups=conv_groups)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, out_channels)
        return x
