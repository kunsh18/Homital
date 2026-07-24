"""
DamageGate — Severe Tissue Damage Features
============================================

Purpose:
  Learn erosion, ulceration, fissure, bleeding, necrosis, open wound.

Design choices:
  • Consumes high-level feature maps + ROI image.
  • Deeper CNN block with spatial attention so the gate can detect
    small but clinically critical regions (e.g. a tiny ulcer).
  • Dark-region and red-region colour cues extracted directly from
    the ROI image and injected as auxiliary spatial hints.
  • Attention bottleneck to focus on high-risk spatial locations
    even when they occupy a small fraction of the lesion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import FeatureGate


class _CBAM_Spatial(nn.Module):
    """Spatial attention from CBAM (Convolutional Block Attention Module)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return x * attn


class DamageGate(FeatureGate):
    """Specialised gate for severe tissue-damage patterns.

    Inputs consumed:
      high      : [B, C_high, h, w]
      roi_image : [B, 3, H, W]
    """

    def __init__(
        self,
        high_channels: int = 256,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)
        inner = 128

        # ── Feature projection ──
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── Colour-cue channels from ROI ──
        # Dark-region mask: pixels that are very dark (potential necrosis)
        # Red-region mask:  pixels with high redness (bleeding, erosion)
        # These are computed from the ROI image and injected as 2 extra channels.
        self.cue_proj = nn.Sequential(
            nn.Conv2d(2, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # ── Deeper CNN block ──
        combined = inner + 16  # 144
        self.deep_block = nn.Sequential(
            nn.Conv2d(combined, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── Spatial attention (CBAM-style) ──
        self.spatial_attn = _CBAM_Spatial(kernel_size=7)

        # ── MLP head ──
        self.mlp = nn.Sequential(
            nn.Linear(inner, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _colour_cues(roi: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Compute dark-region and red-region spatial hints.

        Args:
            roi: [B, 3, H, W] in [0, 1]

        Returns:
            cues: [B, 2, target_h, target_w]
        """
        # Luminance
        lum = 0.299 * roi[:, 0] + 0.587 * roi[:, 1] + 0.114 * roi[:, 2]  # [B, H, W]
        # Dark-region: sigmoid centred at 0.15 brightness
        dark = torch.sigmoid(-(lum - 0.15) * 20).unsqueeze(1)  # [B, 1, H, W]

        # Red-region: high R relative to G and B
        r, g, b = roi[:, 0:1], roi[:, 1:2], roi[:, 2:3]
        redness = (r - (g + b) / 2).clamp(min=0)  # [B, 1, H, W]

        cues = torch.cat([dark, redness], dim=1)  # [B, 2, H, W]
        return F.interpolate(cues, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        high: torch.Tensor = kwargs["high"]         # [B, C_high, h, w]
        roi_image: torch.Tensor = kwargs["roi_image"]  # [B, 3, H, W]

        target_h, target_w = high.shape[2], high.shape[3]

        high_f = self.high_proj(high)   # [B, 128, h, w]

        # Colour cues from ROI
        cues = self._colour_cues(roi_image, target_h, target_w)  # [B, 2, h, w]
        cue_f = self.cue_proj(cues)     # [B, 16, h, w]

        # Concatenate
        cat = torch.cat([high_f, cue_f], dim=1)  # [B, 144, h, w]

        # Deep conv block
        feat = self.deep_block(cat)     # [B, 128, h, w]

        # Spatial attention — focus on small high-risk regions
        feat = self.spatial_attn(feat)  # [B, 128, h, w]

        # Combine global avg + max pooling to retain both average
        # severity and worst-case (max) damage signals
        avg_pool = feat.mean(dim=[2, 3])  # [B, 128]
        max_pool = feat.amax(dim=[2, 3])  # [B, 128]
        pooled = avg_pool + max_pool       # [B, 128]

        return self.mlp(pooled)  # [B, D]
