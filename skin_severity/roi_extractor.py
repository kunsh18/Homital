"""
ROI Extractor — Placeholder for Lesion Detection / Segmentation
================================================================

This module simulates YOLO / U-Net / SAM behaviour by returning:
  • ROI crop   : [B, 3, roi_h, roi_w]
  • Lesion mask : [B, 1, roi_h, roi_w]   (soft mask, 0-1)
  • Bounding box: [B, 4]                  (normalised x1, y1, x2, y2)

**Plug-in guide** ─ replace ``_placeholder_forward`` with one of:
  • YOLOv8 / YOLOv9 detection  → crop + box
  • U-Net segmentation         → mask + crop
  • SAM (Segment Anything)     → mask + crop
  • Any combo of the above

The rest of the pipeline consumes the three outputs above and does
not care how they were produced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIExtractor(nn.Module):
    """Placeholder ROI extractor.

    In production, swap ``forward`` internals with a real detector /
    segmentor.  The interface stays identical:

        roi_crop  : [B, 3, roi_h, roi_w]
        mask      : [B, 1, roi_h, roi_w]
        box_coords: [B, 4]               # normalised (x1, y1, x2, y2)
    """

    def __init__(self, roi_size: int = 224):
        super().__init__()
        self.roi_size = roi_size

        # ── Lightweight learnable "saliency" head (optional) ──
        # This tiny conv head gives the placeholder a *learnable*
        # saliency map so the downstream mask path is exercised
        # during training even before a real segmentor is plugged in.
        self.saliency_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    # ────────────────────────────────────────────────────────
    # PLUG-IN POINT: replace this method with a real detector
    # ────────────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input image tensor [B, 3, H, W]

        Returns:
            roi_crop   : [B, 3, roi_size, roi_size]
            mask       : [B, 1, roi_size, roi_size]  (soft, 0-1)
            box_coords : [B, 4]  normalised (x1, y1, x2, y2)
        """
        B, C, H, W = x.shape

        # ── Placeholder: use the entire image as ROI ──
        roi_crop = F.interpolate(
            x, size=(self.roi_size, self.roi_size), mode="bilinear", align_corners=False
        )  # [B, 3, roi_size, roi_size]

        # ── Generate a soft saliency mask ──
        mask = self.saliency_head(roi_crop)  # [B, 1, roi_size, roi_size]

        # ── Box: full image in normalised coords ──
        box_coords = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0]], device=x.device
        ).expand(B, -1)  # [B, 4]

        return roi_crop, mask, box_coords
