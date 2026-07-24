"""
ShapeBorderGate — Lesion Shape & Border Irregularity Features
==============================================================

Purpose:
  Learn asymmetry, border irregularity, circularity, blurred/jagged edges.

Design choices:
  • Consumes mid + high feature maps, plus the lesion mask (if available).
  • Fixed Sobel-like edge filters + learnable edge-detection convolutions.
  • Spatial attention focused on lesion boundary (derived from mask or
    from feature gradient magnitude when mask is unavailable).
  • Global shape descriptors from mask/box: area ratio, perimeter proxy,
    compactness proxy, bounding-box aspect ratio.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import FeatureGate


class _SobelEdgeConv(nn.Module):
    """Fixed Sobel filters (not learnable) for edge extraction."""

    def __init__(self, in_channels: int):
        super().__init__()
        # Sobel kernels — applied per channel (groups=in_channels)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        sobel_y = sobel_x.transpose(2, 3)

        # Repeat for all input channels → depthwise
        self.register_buffer("sobel_x", sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(in_channels, 1, 1, 1))
        self.groups = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.groups)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.groups)
        return (gx.pow(2) + gy.pow(2)).sqrt()  # gradient magnitude


class _LearnableEdgeConv(nn.Module):
    """Small learnable edge-aware convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ShapeBorderGate(FeatureGate):
    """Specialised gate for shape / border irregularity.

    Inputs consumed:
      mid       : [B, C_mid,  h, w]
      high      : [B, C_high, h, w]
      mask      : [B, 1, H_roi, W_roi]  (optional — may be zeros)
      box       : [B, 4]                 (normalised x1, y1, x2, y2)
    """

    _NUM_SHAPE_STATS = 4  # area_ratio, perimeter_proxy, compactness, aspect_ratio

    def __init__(
        self,
        mid_channels: int = 128,
        high_channels: int = 256,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)
        inner = 64

        # ── Feature projections ──
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

        # ── Edge extraction ──
        self.sobel = _SobelEdgeConv(inner * 2)
        self.learnable_edge = _LearnableEdgeConv(inner * 2, inner)

        # ── Boundary spatial attention ──
        self.boundary_attn_conv = nn.Sequential(
            nn.Conv2d(inner * 2 + 1, 1, 3, padding=1, bias=False),  # +1 for mask/boundary channel
            nn.Sigmoid(),
        )

        # ── MLP head ──
        self.mlp = nn.Sequential(
            nn.Linear(inner + self._NUM_SHAPE_STATS, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _derive_boundary(mask: torch.Tensor, feature_maps: torch.Tensor) -> torch.Tensor:
        """Derive a boundary channel from mask or from feature gradient.

        If the mask is essentially empty (sum < 1), fall back to
        feature-gradient magnitude as a soft boundary proxy.

        Returns: [B, 1, H, W]
        """
        B, _, H, W = feature_maps.shape

        mask_resized = F.interpolate(
            mask, size=(H, W), mode="bilinear", align_corners=False
        )  # [B, 1, H, W]

        has_mask = mask_resized.flatten(1).sum(dim=1) > 1.0  # [B]

        # Boundary from mask: dilate - erode via max-pool / min-pool
        dilated = F.max_pool2d(mask_resized, 3, 1, 1)
        eroded = -F.max_pool2d(-mask_resized, 3, 1, 1)
        mask_boundary = (dilated - eroded).clamp(0, 1)  # [B, 1, H, W]

        # Fallback: gradient magnitude of feature energy
        energy = feature_maps.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        gx = energy[:, :, :, 1:] - energy[:, :, :, :-1]
        gy = energy[:, :, 1:, :] - energy[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        grad_boundary = (gx.pow(2) + gy.pow(2)).sqrt()
        grad_boundary = grad_boundary / (grad_boundary.amax(dim=[2, 3], keepdim=True) + 1e-6)

        # Per-sample selection
        has_mask_view = has_mask.float().view(B, 1, 1, 1)
        boundary = has_mask_view * mask_boundary + (1 - has_mask_view) * grad_boundary
        return boundary  # [B, 1, H, W]

    @staticmethod
    def _shape_stats(mask: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Compute global shape descriptors.

        Returns: [B, 4]   (area_ratio, perimeter_proxy, compactness, aspect_ratio)
        """
        B = mask.shape[0]

        # Area ratio: fraction of ROI covered by lesion
        area_ratio = mask.flatten(1).mean(dim=1, keepdim=True)  # [B, 1]

        # Perimeter proxy: sum of boundary pixels
        dilated = F.max_pool2d(mask, 3, 1, 1)
        eroded = -F.max_pool2d(-mask, 3, 1, 1)
        perim = (dilated - eroded).clamp(0, 1).flatten(1).sum(dim=1, keepdim=True)
        # Normalise by image diagonal
        _, _, H, W = mask.shape
        diag = math.sqrt(H * H + W * W) + 1e-6
        perim_norm = perim / diag  # [B, 1]

        # Compactness proxy: 4π·area / perimeter²
        area_px = mask.flatten(1).sum(dim=1, keepdim=True)
        compactness = (4 * math.pi * area_px) / (perim.pow(2) + 1e-6)  # [B, 1]

        # Aspect ratio from bounding box
        bw = (box[:, 2] - box[:, 0]).unsqueeze(1).clamp(min=1e-6)  # [B, 1]
        bh = (box[:, 3] - box[:, 1]).unsqueeze(1).clamp(min=1e-6)
        aspect = bw / bh  # [B, 1]

        return torch.cat([area_ratio, perim_norm, compactness, aspect], dim=1)  # [B, 4]

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        mid: torch.Tensor = kwargs["mid"]     # [B, C_mid, h2, w2]
        high: torch.Tensor = kwargs["high"]   # [B, C_high, h3, w3]
        mask: torch.Tensor = kwargs["mask"]   # [B, 1, H, W]
        box: torch.Tensor = kwargs["box"]     # [B, 4]

        target_h, target_w = mid.shape[2], mid.shape[3]

        mid_f = self.mid_proj(mid)  # [B, 64, h, w]
        high_up = F.interpolate(high, size=(target_h, target_w), mode="bilinear", align_corners=False)
        high_f = self.high_proj(high_up)  # [B, 64, h, w]

        cat = torch.cat([mid_f, high_f], dim=1)  # [B, 128, h, w]

        # Sobel edge features
        sobel_feat = self.sobel(cat)              # [B, 128, h, w]

        # Learnable edge convolution
        learn_feat = self.learnable_edge(cat)     # [B, 64, h, w]

        # Derive boundary attention map
        boundary = self._derive_boundary(mask, cat)  # [B, 1, h, w]

        # Spatial attention using sobel + boundary
        attn_input = torch.cat([sobel_feat, boundary], dim=1)  # [B, 129, h, w]
        attn_map = self.boundary_attn_conv(attn_input)         # [B, 1, h, w]

        # Apply attention to learned edge features
        attended = learn_feat * attn_map  # [B, 64, h, w]

        pooled = attended.mean(dim=[2, 3])  # [B, 64]

        # Shape statistics
        stats = self._shape_stats(mask, box)  # [B, 4]

        return self.mlp(torch.cat([pooled, stats], dim=1))  # [B, D]
