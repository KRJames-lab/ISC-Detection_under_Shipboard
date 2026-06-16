"""
ModernTCN — Modern Pure Convolution for Time Series (ICLR 2024 Spotlight)
Lightweight variant for STM32N6 NPU deployment.
Input: (batch, 2, 1000) at 100Hz — 10s window
Reference: Luo et al., "ModernTCN: A Modern Pure Convolution Structure
           for General Time Series Analysis", ICLR 2024.
"""
import torch
import torch.nn as nn
from ai.config import (
    INPUT_CHANNELS, MODERN_TCN_D_MODEL, MODERN_TCN_STEM_KERNEL,
    MODERN_TCN_KERNEL, MODERN_TCN_BLOCKS, MODERN_TCN_FFN_RATIO,
)


class ModernTCNBlock(nn.Module):
    """DWConv (temporal mixing) + ConvFFN (channel mixing) with residual."""

    def __init__(self, d_model, kernel_size, ffn_ratio):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.norm = nn.BatchNorm1d(d_model)
        ffn_hidden = d_model * ffn_ratio
        self.pw1 = nn.Conv1d(d_model, ffn_hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(ffn_hidden, d_model, 1)

    def forward(self, x):
        residual = x
        x = self.norm(self.dw_conv(x))
        x = self.pw2(self.act(self.pw1(x)))
        return x + residual


class ModernTCN(nn.Module):
    def __init__(self):
        super().__init__()
        d = MODERN_TCN_D_MODEL
        sk = MODERN_TCN_STEM_KERNEL
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, d, kernel_size=sk, padding=sk // 2),
            nn.BatchNorm1d(d),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ModernTCNBlock(d, MODERN_TCN_KERNEL, MODERN_TCN_FFN_RATIO)
            for _ in range(MODERN_TCN_BLOCKS)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
