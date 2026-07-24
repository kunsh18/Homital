"""
TextureGate — Surface Texture Severity Features
=================================================

Purpose:
  Learn scaling, crusting, dryness, roughness, flaking patterns.

Design choices:
  • Consumes low + mid feature maps (texture is a low/mid-level cue).
  • Multi-kernel depthwise-separable branches (3×3, 5×5) to capture
    fine and coarse surface texture at different scales.
  • Local spatial attention to emphasise rough / textured regions.
  • Top-k spatial activation selection — an architectural sparsity
    mechanism (not a loss) that forces the gate to attend to the
    most textured patches rather than averaging over smooth skin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import FeatureGate


class _DepthwiseSeparable(nn.Module):
    """Depthwise-separable conv block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn2(self.pw(self.relu(self.bn1(self.dw(x))))))


class _SpatialAttention(nn.Module):
    """Lightweight spatial attention (max + mean channel squeeze)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        mx = x.amax(dim=1, keepdim=True)   # [B, 1, H, W]
        attn = self.conv(torch.cat([avg, mx], dim=1))  # [B, 1, H, W]
        return x * attn


class TextureGate(FeatureGate):
    """Specialised gate for surface-texture severity cues.

    Inputs consumed:
      low : [B, C_low,  h1, w1]
      mid : [B, C_mid,  h2, w2]
    """

    def __init__(
        self,
        low_channels: int = 64,
        mid_channels: int = 128,
        embed_dim: int = 128,
        topk_ratio: float = 0.25,
    ):
        super().__init__(embed_dim=embed_dim)
        self.topk_ratio = topk_ratio

        inner = 64  # internal channel width

        # ── Project low / mid to the same channel count ──
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── Multi-kernel depthwise-separable branches ──
        self.branch3 = _DepthwiseSeparable(inner * 2, inner, 3)  # fine texture
        self.branch5 = _DepthwiseSeparable(inner * 2, inner, 5)  # coarse texture

        # ── Spatial attention ──
        self.spatial_attn = _SpatialAttention()

        # ── MLP head ──
        self.mlp = nn.Sequential(
            nn.Linear(inner * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        low: torch.Tensor = kwargs["low"]   # [B, C_low, h1, w1]
        mid: torch.Tensor = kwargs["mid"]   # [B, C_mid, h2, w2]

        target_h, target_w = low.shape[2], low.shape[3]

        low_f = self.low_proj(low)            # [B, 64, h, w]
        mid_up = F.interpolate(mid, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mid_f = self.mid_proj(mid_up)         # [B, 64, h, w]

        cat = torch.cat([low_f, mid_f], dim=1)  # [B, 128, h, w]

        # Multi-kernel texture branches
        t3 = self.branch3(cat)                # [B, 64, h, w]
        t5 = self.branch5(cat)                # [B, 64, h, w]
        fused = torch.cat([t3, t5], dim=1)    # [B, 128, h, w]

        # Spatial attention to highlight rough patches
        fused = self.spatial_attn(fused)      # [B, 128, h, w]

        # ── Top-k spatial activation (architectural sparsity) ──
        B, C, H, W = fused.shape
        flat = fused.flatten(2)               # [B, 128, H*W]
        energy = flat.mean(dim=1)             # [B, H*W]
        k = max(1, int(H * W * self.topk_ratio))
        topk_vals, topk_idx = energy.topk(k, dim=1)  # [B, k]

        # Gather top-k spatial positions
        topk_idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]
        selected = flat.gather(2, topk_idx_exp)       # [B, C, k]
        pooled = selected.mean(dim=2)                  # [B, 128]

        return self.mlp(pooled)  # [B, D]
