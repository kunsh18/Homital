"""
Shared Visual Backbone — Multi-Scale Feature Extraction
========================================================

Extracts three feature-map levels from the ROI crop:

  low  : [B, C_low,  H/4,  W/4 ]  — edges, fine texture, local colour
  mid  : [B, C_mid,  H/8,  W/8 ]  — scaling, roughness, crusting
  high : [B, C_high, H/16, W/16]  — lesion morphology, damage patterns

The backbone uses grouped residual blocks so that it remains
lightweight enough to train on a single GPU.
"""

import torch
import torch.nn as nn


# ─────────────────────── building blocks ───────────────────────


class ConvBNReLU(nn.Module):
    """Conv → BatchNorm → ReLU convenience block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Simple pre-activation residual block with optional down-sample."""

    def __init__(self, channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = ConvBNReLU(channels, channels, 3, stride, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(channels),
            )
            if downsample
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


# ─────────────────────── backbone ─────────────────────────────


class SharedBackbone(nn.Module):
    """Lightweight CNN backbone producing low / mid / high feature maps.

    Architecture
    ────────────
    stem   : 3 → 64,  stride-2  →  H/2, W/2
    stage1 : 64 → 64,  stride-2  →  H/4, W/4   → **low**   (C=64)
    stage2 : 64 → 128, stride-2  →  H/8, W/8   → **mid**   (C=128)
    stage3 : 128→ 256, stride-2  →  H/16, W/16  → **high**  (C=256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: tuple[int, int, int] = (2, 3, 3),
    ):
        super().__init__()

        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256

        # Stem: 3 → c1, spatial /2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )  # → [B, 64, H/2, W/2]

        # Stage 1 → low-level features  (edges, fine texture, local colour)
        self.stage1 = self._make_stage(c1, c1, num_blocks[0], downsample_first=True)
        # → [B, 64, H/4, W/4]

        # Channel expansion 64 → 128
        self.expand2 = ConvBNReLU(c1, c2, 1, 1, 0)

        # Stage 2 → mid-level features  (scaling, roughness, crusting)
        self.stage2 = self._make_stage(c2, c2, num_blocks[1], downsample_first=True)
        # → [B, 128, H/8, W/8]

        # Channel expansion 128 → 256
        self.expand3 = ConvBNReLU(c2, c3, 1, 1, 0)

        # Stage 3 → high-level features (morphology, damage, severity)
        self.stage3 = self._make_stage(c3, c3, num_blocks[2], downsample_first=True)
        # → [B, 256, H/16, W/16]

        # Expose channel dims so gates can introspect
        self.low_channels = c1    # 64
        self.mid_channels = c2    # 128
        self.high_channels = c3   # 256

    # ── helpers ──

    @staticmethod
    def _make_stage(
        in_ch: int, out_ch: int, n_blocks: int, downsample_first: bool
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        if in_ch != out_ch:
            layers.append(ConvBNReLU(in_ch, out_ch, 1, 1, 0))
        for i in range(n_blocks):
            layers.append(ResBlock(out_ch, downsample=(i == 0 and downsample_first)))
        return nn.Sequential(*layers)

    # ── forward ──

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ROI crop [B, 3, H, W]

        Returns:
            low  : [B,  64, H/4,  W/4 ]
            mid  : [B, 128, H/8,  W/8 ]
            high : [B, 256, H/16, W/16]
        """
        s = self.stem(x)               # [B,  64, H/2,  W/2]
        low = self.stage1(s)            # [B,  64, H/4,  W/4]
        mid = self.stage2(self.expand2(low))   # [B, 128, H/8,  W/8]
        high = self.stage3(self.expand3(mid))  # [B, 256, H/16, W/16]
        return low, mid, high
