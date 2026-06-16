"""
LITE — Light Inception with boosTing tEchniques (IEEE DSAA 2023)
Simplified variant: no custom filters, 2-channel input.
Input: (batch, 2, 1000) at 100Hz — 10s window
Reference: Ali et al., "LITE: Light Inception with boosTing tEchniques
           for Time Series Classification", IEEE DSAA 2023.
"""
import torch
import torch.nn as nn
from ai.config import (
    INPUT_CHANNELS, LITE_INCEPTION_KERNELS,
    LITE_INCEPTION_FILTERS, LITE_DWS_CHANNELS,
    LITE_DWS1_KERNEL, LITE_DWS2_KERNEL,
)


class InceptionLayer(nn.Module):
    """Multi-scale parallel Conv1d (Inception-style)."""

    def __init__(self, in_channels, n_filters, kernel_sizes):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, n_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        total_ch = n_filters * len(kernel_sizes)
        self.norm = nn.BatchNorm1d(total_ch)
        self.act = nn.GELU()

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        return self.act(self.norm(x))


class DWSConvLayer(nn.Module):
    """Depthwise Separable Conv1d block."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels,
        )
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.pw = nn.Conv1d(in_channels, out_channels, 1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.dw(x)))
        x = self.act(self.norm2(self.pw(x)))
        return x


class LITE(nn.Module):
    def __init__(self):
        super().__init__()
        n_branches = len(LITE_INCEPTION_KERNELS)
        inception_out = LITE_INCEPTION_FILTERS * n_branches

        self.inception = InceptionLayer(
            INPUT_CHANNELS, LITE_INCEPTION_FILTERS, LITE_INCEPTION_KERNELS,
        )
        self.dws1 = DWSConvLayer(inception_out, LITE_DWS_CHANNELS, kernel_size=LITE_DWS1_KERNEL)
        self.dws2 = DWSConvLayer(LITE_DWS_CHANNELS, LITE_DWS_CHANNELS, kernel_size=LITE_DWS2_KERNEL)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(LITE_DWS_CHANNELS, 1),
        )

    def forward(self, x):
        x = self.inception(x)
        x = self.dws1(x)
        x = self.dws2(x)
        x = self.head(x)
        return x.squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
