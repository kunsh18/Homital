"""
MorphologyGate — High-Level Lesion Morphology Features
========================================================

Purpose:
  Learn lesion morphology such as macule, papule, plaque, pustule,
  vesicle, nodule, ulcer-like structure.

Design choices:
  • Consumes mid + high feature maps (morphology is a semantic cue).
  • Deeper residual blocks with dilated convolutions to capture wider
    lesion structure without losing resolution.
  • Global average pooling for lesion-level morphology embedding.
  • This gate is intentionally *more semantic* and deeper than the
    colour / texture gates, encouraging it to learn abstract shape
    and structure rather than low-level patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import FeatureGate


class _DilatedResBlock(nn.Module):
    """Residual block with dilated convolutions for wider receptive field."""

    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class MorphologyGate(FeatureGate):
    """Specialised gate for lesion morphology classification cues.

    Inputs consumed:
      mid  : [B, C_mid,  h, w]
      high : [B, C_high, h, w]
    """

    def __init__(
        self,
        mid_channels: int = 128,
        high_channels: int = 256,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)
        inner = 128

        # ── Project mid + high to shared channel count ──
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── Deeper dilated residual stack ──
        # Dilation rates 1, 2, 4 give an effective receptive field
        # covering most of the lesion in the feature map.
        self.res_stack = nn.Sequential(
            _DilatedResBlock(inner * 2, dilation=1),
            _DilatedResBlock(inner * 2, dilation=2),
            _DilatedResBlock(inner * 2, dilation=4),
        )

        # ── Channel reduction ──
        self.reduce = nn.Sequential(
            nn.Conv2d(inner * 2, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── MLP head ──
        self.mlp = nn.Sequential(
            nn.Linear(inner, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        mid: torch.Tensor = kwargs["mid"]     # [B, C_mid, h2, w2]
        high: torch.Tensor = kwargs["high"]   # [B, C_high, h3, w3]

        target_h, target_w = mid.shape[2], mid.shape[3]

        mid_f = self.mid_proj(mid)  # [B, 128, h, w]
        high_up = F.interpolate(high, size=(target_h, target_w), mode="bilinear", align_corners=False)
        high_f = self.high_proj(high_up)  # [B, 128, h, w]

        cat = torch.cat([mid_f, high_f], dim=1)  # [B, 256, h, w]

        # Deep dilated residual processing
        feat = self.res_stack(cat)     # [B, 256, h, w]
        feat = self.reduce(feat)       # [B, 128, h, w]

        # Global average pooling → morphology embedding
        pooled = feat.mean(dim=[2, 3])  # [B, 128]

        return self.mlp(pooled)  # [B, D]
