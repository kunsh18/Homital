"""
ColorGate — Chromatic Severity Features
=========================================

Purpose:
  Learn redness / erythema, pigmentation changes, colour variance,
  yellow-crust regions, dark / blue / black tones.

Design choices:
  • Consumes the ROI image + low + mid feature maps.
  • 1×1 convolutions for channel-wise colour mixing (no spatial bias).
  • Global colour statistics from the ROI: mean/std RGB, approximate
    redness index, brightness, saturation proxy.
  • SE-style channel attention to let the gate focus on the most
    colour-informative channels.
  • Deliberately avoids deep spatial convolutions — colour cues are
    chromatic, not spatial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import FeatureGate


class _ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, _, _ = x.shape
        s = x.mean(dim=[2, 3])          # [B, C]   — squeeze
        w = self.fc(s).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w                     # excite


class ColorGate(FeatureGate):
    """Specialised gate for chromatic / colour-based severity features.

    Inputs consumed:
      roi_image : [B, 3, H, W]    — ROI crop (original pixels)
      low       : [B, C_low,  h, w] — low-level feature map
      mid       : [B, C_mid,  h, w] — mid-level feature map
    """

    # Number of hand-crafted colour statistics extracted from the ROI
    _NUM_COLOR_STATS = 10  # mean_rgb(3) + std_rgb(3) + redness(1) + brightness(1) + sat(1) + contrast(1)

    def __init__(
        self,
        low_channels: int = 64,
        mid_channels: int = 128,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)

        # ── 1×1 convolutions on ROI image for channel-wise colour mixing ──
        self.color_conv = nn.Sequential(
            nn.Conv2d(3, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── 1×1 projections to merge low + mid features ──
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Channel attention over the fused colour feature map ──
        fused_ch = 32 + 32 + 32  # color_conv + low_proj + mid_proj = 96
        self.channel_attn = _ChannelAttention(fused_ch, reduction=4)

        # ── Global pooling → MLP ──
        self.mlp = nn.Sequential(
            nn.Linear(fused_ch + self._NUM_COLOR_STATS, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    # ── colour statistics ──

    @staticmethod
    def _colour_stats(roi: torch.Tensor) -> torch.Tensor:
        """Extract global colour statistics from the ROI image.

        Args:
            roi: [B, 3, H, W] in [0, 1] range

        Returns:
            stats: [B, 10]
        """
        B = roi.shape[0]
        # Flatten spatial dims → [B, 3, N]
        flat = roi.flatten(2)

        mean_rgb = flat.mean(dim=2)                            # [B, 3]
        std_rgb = flat.std(dim=2).clamp(min=1e-6)              # [B, 3]

        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

        redness = (r - (g + b) / 2).unsqueeze(1)               # [B, 1]
        brightness = mean_rgb.mean(dim=1, keepdim=True)         # [B, 1]
        # Saturation proxy: max(RGB) - min(RGB) per pixel, then average
        sat = (roi.amax(dim=1) - roi.amin(dim=1)).flatten(1).mean(dim=1, keepdim=True)  # [B, 1]
        # Contrast proxy: std of luminance
        lum = 0.299 * roi[:, 0] + 0.587 * roi[:, 1] + 0.114 * roi[:, 2]  # [B, H, W]
        contrast = lum.flatten(1).std(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]

        return torch.cat([mean_rgb, std_rgb, redness, brightness, sat, contrast], dim=1)  # [B, 10]

    # ── gate forward ──

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        roi_image: torch.Tensor = kwargs["roi_image"]  # [B, 3, H, W]
        low: torch.Tensor = kwargs["low"]               # [B, C_low, h1, w1]
        mid: torch.Tensor = kwargs["mid"]               # [B, C_mid, h2, w2]

        # Target spatial size: use low-level map size
        target_h, target_w = low.shape[2], low.shape[3]

        # 1×1 colour convolutions on ROI
        roi_down = F.interpolate(
            roi_image, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        color_feat = self.color_conv(roi_down)       # [B, 32, h, w]

        # Project low / mid to 32 channels each, align spatially
        low_feat = self.low_proj(low)                 # [B, 32, h, w]
        mid_up = F.interpolate(mid, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mid_feat = self.mid_proj(mid_up)              # [B, 32, h, w]

        # Fuse along channel dim
        fused = torch.cat([color_feat, low_feat, mid_feat], dim=1)  # [B, 96, h, w]

        # Channel attention
        fused = self.channel_attn(fused)              # [B, 96, h, w]

        # Global average pool
        pooled = fused.mean(dim=[2, 3])               # [B, 96]

        # Colour statistics
        stats = self._colour_stats(roi_image)         # [B, 10]

        # MLP → embedding
        return self.mlp(torch.cat([pooled, stats], dim=1))  # [B, D]
