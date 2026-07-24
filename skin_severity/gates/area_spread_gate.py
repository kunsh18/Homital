"""
AreaSpreadGate — Lesion Size, Spread & Density Features
========================================================

Purpose:
  Learn lesion size, affected percentage, lesion count proxy,
  density, and spatial spread.

Design choices:
  • Consumes ROI mask, bounding-box coordinates, and mid-level
    feature maps.
  • Feature-map spatial activation statistics: activated area ratio,
    average activation, max activation, spatial entropy-like spread.
  • Simple MLP on geometric features: box width, height, area,
    aspect ratio.
  • When mask is unavailable, spread is estimated from feature-map
    activations alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import FeatureGate


class AreaSpreadGate(FeatureGate):
    """Specialised gate for lesion area / spread / density.

    Inputs consumed:
      mid  : [B, C_mid, h, w]
      mask : [B, 1, H, W]       (soft, 0-1 — may be zeros)
      box  : [B, 4]             (normalised x1, y1, x2, y2)
    """

    # Feature-map statistics (8) + geometric features (4)
    _NUM_FEATURES = 12

    def __init__(
        self,
        mid_channels: int = 128,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)

        # ── 1×1 conv to compress feature map for activation stats ──
        self.compress = nn.Sequential(
            nn.Conv2d(mid_channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # ── MLP on combined statistics ──
        self.mlp = nn.Sequential(
            nn.Linear(self._NUM_FEATURES, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _spatial_stats(act_map: torch.Tensor) -> torch.Tensor:
        """Compute spatial activation statistics from activation map.

        Args:
            act_map: [B, 1, h, w]  (0-1 after sigmoid)

        Returns:
            stats: [B, 8]
        """
        flat = act_map.flatten(1)  # [B, h*w]

        # 1. Activated area ratio (fraction of pixels > 0.5)
        activated_ratio = (flat > 0.5).float().mean(dim=1, keepdim=True)  # [B, 1]

        # 2. Average activation
        avg_act = flat.mean(dim=1, keepdim=True)  # [B, 1]

        # 3. Max activation
        max_act = flat.amax(dim=1, keepdim=True)  # [B, 1]

        # 4. Std of activation (spread indicator)
        std_act = flat.std(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]

        # 5. Spatial entropy proxy: -sum(p * log(p))
        p = flat / (flat.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(p * (p + 1e-8).log()).sum(dim=1, keepdim=True)
        # Normalise by max entropy (log N)
        import math
        max_ent = math.log(flat.shape[1] + 1e-8)
        entropy = entropy / max_ent  # [B, 1]

        # 6. Skewness of activation
        mean_val = flat.mean(dim=1, keepdim=True)
        diff = flat - mean_val
        skewness = (diff.pow(3).mean(dim=1, keepdim=True)) / (std_act.pow(3) + 1e-8)  # [B, 1]

        # 7. Kurtosis of activation
        kurtosis = (diff.pow(4).mean(dim=1, keepdim=True)) / (std_act.pow(4) + 1e-8) - 3.0  # [B, 1]

        # 8. Median activation
        median_act = flat.median(dim=1, keepdim=True).values  # [B, 1]

        return torch.cat([
            activated_ratio, avg_act, max_act, std_act,
            entropy, skewness, kurtosis, median_act
        ], dim=1)  # [B, 8]

    @staticmethod
    def _geometric_features(box: torch.Tensor) -> torch.Tensor:
        """Compute geometric features from normalised bounding box.

        Args:
            box: [B, 4]  (x1, y1, x2, y2)  normalised 0-1

        Returns:
            geo: [B, 4]  (width, height, area, aspect_ratio)
        """
        w = (box[:, 2] - box[:, 0]).clamp(min=1e-6).unsqueeze(1)  # [B, 1]
        h = (box[:, 3] - box[:, 1]).clamp(min=1e-6).unsqueeze(1)
        area = w * h
        aspect = w / h
        return torch.cat([w, h, area, aspect], dim=1)  # [B, 4]

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        mid: torch.Tensor = kwargs["mid"]    # [B, C_mid, h, w]
        mask: torch.Tensor = kwargs["mask"]  # [B, 1, H, W]
        box: torch.Tensor = kwargs["box"]    # [B, 4]

        # Compress feature map to 1-channel activation map
        act_map = self.compress(mid)  # [B, 1, h, w]

        # If mask has content, modulate the activation map
        has_mask = mask.flatten(1).sum(dim=1) > 1.0  # [B]
        if has_mask.any():
            mask_down = F.interpolate(
                mask, size=act_map.shape[2:], mode="bilinear", align_corners=False
            )
            # Blend: use mask where available, activation map otherwise
            blend = has_mask.float().view(-1, 1, 1, 1)
            act_map = blend * (act_map * mask_down) + (1 - blend) * act_map

        # Spatial statistics
        sp_stats = self._spatial_stats(act_map)     # [B, 8]

        # Geometric features
        geo = self._geometric_features(box)          # [B, 4]

        combined = torch.cat([sp_stats, geo], dim=1)  # [B, 12]
        return self.mlp(combined)  # [B, D]
