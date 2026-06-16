"""
NPU-Conv2D — 2D Conv model for STM32N6 Neural-ART NPU deployment.

Reshapes 1D time series (2, 1000) into 2D grid (2, 20, 50) and applies
MobileNetV2-style inverted residual blocks with depthwise separable Conv2d.

2D factorization rationale:
  - Horizontal (W=50, 0.5s): captures vibration patterns (4-33Hz MIL-STD)
  - Vertical (H=20, segments): captures ISC voltage drift across 10s window

All operations NPU-compatible:
  - Conv2d kernel <= 5x5 (NPU limit: 6x6 at stride=1)
  - Stride in {1, 2}
  - W x C <= 2048 at every layer (verified)
  - DWConv2d exempt from W x C limit
  - ReLU activation (NPU HW accelerated)

Input:  (batch, 2, 1000) at 100Hz - 10s window
Output: (batch, 1) logit (>0 = ISC detected)

Reference architecture: MobileNetV2 (Sandler et al., CVPR 2018)
"""
import torch
import torch.nn as nn


# 2D reshape dimensions
RESHAPE_H = 20   # 20 segments (0.5s each)
RESHAPE_W = 50   # 50 samples per segment


class InvertedResidual2D(nn.Module):
    """MobileNetV2 inverted residual: expand -> DWConv -> project.

    All kernels 3x3, stride in {1, 2}. NPU HW compatible.
    """

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.conv = nn.Sequential(
            # Expand: pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            # Depthwise: 3x3
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                      padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            # Project: pointwise (no activation)
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class NPUConv2D(nn.Module):
    """2D Conv model for NPU deployment.

    Architecture (NPU constraint verified at each layer):
        Reshape:  (B, 2, 1000) -> (B, 2, 20, 50)
        Stem:     Conv2d(2->16, 3x3)       -> (B, 16, 20, 50)  W*C=800
        Block1:   InvRes(16->32, s=2)       -> (B, 32, 10, 25)  W*C=800
        Block2:   InvRes(32->32, s=1) +res  -> (B, 32, 10, 25)
        Block3:   InvRes(32->64, s=2)       -> (B, 64, 5, 13)   W*C=832
        Block4:   InvRes(64->64, s=1) +res  -> (B, 64, 5, 13)
        Block5:   InvRes(64->64, s=2)       -> (B, 64, 3, 7)    W*C=448
        GAP -> Linear(64, 1)
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            InvertedResidual2D(16, 32, stride=2, expand_ratio=2),
            InvertedResidual2D(32, 32, stride=1, expand_ratio=2),
            InvertedResidual2D(32, 64, stride=2, expand_ratio=2),
            InvertedResidual2D(64, 64, stride=1, expand_ratio=2),
            InvertedResidual2D(64, 64, stride=2, expand_ratio=2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        # 1D -> 2D: (B, 2, 1000) -> (B, 2, 20, 50)
        x = x.view(B, 2, RESHAPE_H, RESHAPE_W)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
