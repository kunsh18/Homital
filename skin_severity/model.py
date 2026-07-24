"""
SkinSeverityModel — Top-Level Gated Architecture
==================================================

Wires together:
  ROIExtractor → SharedBackbone → 8 Specialised Gates → GatedFusion

Predicts a continuous clinical severity score in [0.0, 3.0].
Returns severity score, gate weights, interpreted severity band,
and an explanation dictionary showing per-gate contributions.
"""

import torch
import torch.nn as nn

from .roi_extractor import ROIExtractor
from .backbone import SharedBackbone
from .gates import (
    ColorGate,
    TextureGate,
    ShapeBorderGate,
    MorphologyGate,
    DamageGate,
    AreaSpreadGate,
    LocationRiskGate,
    MetadataGate,
)
from .fusion import GatedFusion, interpret_severity


# Gate names — order must match the list built in __init__
GATE_NAMES = [
    "color",
    "texture",
    "shape_border",
    "morphology",
    "damage",
    "area_spread",
    "location_risk",
    "metadata",
]


class SkinSeverityModel(nn.Module):
    """End-to-end gated severity-scoring model for skin-disease images.

    Outputs a continuous severity score ∈ [0.0, 3.0]:
      0.0 – 0.5  → very mild
      0.5 – 1.2  → mild
      1.2 – 2.0  → moderate
      2.0 – 2.6  → severe
      2.6 – 3.0  → critical / extreme

    Parameters
    ----------
    roi_size : int
        Spatial resolution the ROI extractor resizes crops to.
    base_channels : int
        Base channel width of the shared backbone.
    embed_dim : int
        Dimensionality of each gate's output embedding.
    metadata_dim : int
        Number of metadata features expected in the metadata tensor.
    num_locations : int
        Number of discrete body-location categories.
    """

    def __init__(
        self,
        roi_size: int = 224,
        base_channels: int = 64,
        embed_dim: int = 128,
        metadata_dim: int = 10,
        num_locations: int = 20,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ──────────── Stage 1: ROI Extraction ────────────
        self.roi_extractor = ROIExtractor(roi_size=roi_size)

        # ──────────── Stage 2: Shared Backbone ────────────
        self.backbone = SharedBackbone(
            in_channels=3, base_channels=base_channels
        )

        low_ch = self.backbone.low_channels    # 64
        mid_ch = self.backbone.mid_channels    # 128
        high_ch = self.backbone.high_channels  # 256

        # ──────────── Stage 3: Specialised Gates ────────────
        self.color_gate = ColorGate(
            low_channels=low_ch, mid_channels=mid_ch, embed_dim=embed_dim
        )
        self.texture_gate = TextureGate(
            low_channels=low_ch, mid_channels=mid_ch, embed_dim=embed_dim
        )
        self.shape_border_gate = ShapeBorderGate(
            mid_channels=mid_ch, high_channels=high_ch, embed_dim=embed_dim
        )
        self.morphology_gate = MorphologyGate(
            mid_channels=mid_ch, high_channels=high_ch, embed_dim=embed_dim
        )
        self.damage_gate = DamageGate(
            high_channels=high_ch, embed_dim=embed_dim
        )
        self.area_spread_gate = AreaSpreadGate(
            mid_channels=mid_ch, embed_dim=embed_dim
        )
        self.location_risk_gate = LocationRiskGate(
            num_locations=num_locations, embed_dim=embed_dim
        )
        self.metadata_gate = MetadataGate(
            metadata_dim=metadata_dim, embed_dim=embed_dim
        )

        # Ordered list of gates (must match GATE_NAMES)
        self._gates = nn.ModuleList([
            self.color_gate,
            self.texture_gate,
            self.shape_border_gate,
            self.morphology_gate,
            self.damage_gate,
            self.area_spread_gate,
            self.location_risk_gate,
            self.metadata_gate,
        ])

        # ──────────── Stage 4: Gated Fusion ────────────
        self.fusion = GatedFusion(
            num_gates=len(self._gates),
            embed_dim=embed_dim,
        )

    # ────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None = None,
        location: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Args:
            image    : [B, 3, H, W]  skin image (values in [0, 1])
            metadata : [B, M]        optional patient metadata
            location : [B]           optional body-location index (-1 = missing)

        Returns:
            dict with keys:
              'severity_score' : [B, 1]  continuous score in [0.0, 3.0]
              'severity_bands' : list[str]  interpreted band per sample
              'gate_weights'   : [B, 8]
              'fused_embedding': [B, D]
              'explanations'   : dict mapping gate name → contribution [B, 1]
        """
        B = image.shape[0]

        # ───── 1. ROI Extraction ─────
        roi_image, mask, box = self.roi_extractor(image)
        # roi_image : [B, 3, roi_h, roi_w]
        # mask      : [B, 1, roi_h, roi_w]
        # box       : [B, 4]

        # ───── 2. Shared Backbone ─────
        low, mid, high = self.backbone(roi_image)
        # low  : [B,  64, roi_h/4,  roi_w/4 ]
        # mid  : [B, 128, roi_h/8,  roi_w/8 ]
        # high : [B, 256, roi_h/16, roi_w/16]

        # ───── 3. Gate Forward Passes ─────
        # Each gate receives only the inputs relevant to its purpose.

        gate_embeddings: list[torch.Tensor] = []
        gate_scores: list[torch.Tensor] = []

        # A. ColorGate — ROI image + low + mid
        emb, sc = self.color_gate(roi_image=roi_image, low=low, mid=mid)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # B. TextureGate — low + mid
        emb, sc = self.texture_gate(low=low, mid=mid)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # C. ShapeBorderGate — mid + high + mask + box
        emb, sc = self.shape_border_gate(mid=mid, high=high, mask=mask, box=box)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # D. MorphologyGate — mid + high
        emb, sc = self.morphology_gate(mid=mid, high=high)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # E. DamageGate — high + ROI image
        emb, sc = self.damage_gate(high=high, roi_image=roi_image)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # F. AreaSpreadGate — mid + mask + box
        emb, sc = self.area_spread_gate(mid=mid, mask=mask, box=box)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # G. LocationRiskGate — location index
        emb, sc = self.location_risk_gate(location=location, _batch_size=B)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # H. MetadataGate — metadata tensor
        emb, sc = self.metadata_gate(metadata=metadata, _batch_size=B)
        gate_embeddings.append(emb)
        gate_scores.append(sc)

        # ───── 4. Gated Fusion ─────
        fusion_out = self.fusion(gate_embeddings, gate_scores)
        # fusion_out['severity']     : [B, 1]  ∈ [0.0, 3.0]
        # fusion_out['gate_weights'] : [B, 8]

        # ───── 5. Build Explanation Dict ─────
        gate_weights = fusion_out["gate_weights"]  # [B, 8]
        explanations: dict[str, torch.Tensor] = {}
        for i, name in enumerate(GATE_NAMES):
            explanations[name] = gate_weights[:, i : i + 1]  # [B, 1]

        # ───── 6. Interpret Severity Bands (inference only) ─────
        severity_score = fusion_out["severity"]  # [B, 1]
        severity_bands = [
            interpret_severity(s.item()) for s in severity_score
        ]

        return {
            "severity_score": severity_score,
            "severity_bands": severity_bands,
            "gate_weights": gate_weights,
            "fused_embedding": fusion_out["fused"],
            "explanations": explanations,
        }
