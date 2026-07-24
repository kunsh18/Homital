"""
LocationRiskGate — Anatomical Location Risk Features
======================================================

Purpose:
  Encode the clinical risk associated with the body location of the
  lesion (face, eyes, lips, genitals, palms/soles, scalp, nails, …).

Design choices:
  • Consumes a location index from metadata.
  • Learnable embedding table over discrete body locations.
  • Small MLP with dropout.
  • Returns a zero embedding when metadata is missing (location == -1
    or not provided), so the gate gracefully degrades.
"""

import torch
import torch.nn as nn

from .base_gate import FeatureGate


class LocationRiskGate(FeatureGate):
    """Specialised gate for anatomical-location risk.

    Inputs consumed:
      location : [B]   (integer index, -1 = missing)
    """

    def __init__(
        self,
        num_locations: int = 20,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)
        self.num_locations = num_locations

        # ── Learnable location embedding ──
        # Index 0 is reserved for "unknown / missing".
        self.location_emb = nn.Embedding(num_locations + 1, 64, padding_idx=0)

        # ── MLP ──
        self.mlp = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
        )

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        location: torch.Tensor | None = kwargs.get("location", None)

        if location is None:
            B = kwargs.get("_batch_size", 1)
            device = next(self.parameters()).device
            return torch.zeros(B, self.embed_dim, device=device)

        # Map -1 (missing) → 0 (padding_idx), valid indices → idx+1
        safe_loc = location.clamp(min=-1)
        idx = (safe_loc + 1).clamp(min=0, max=self.num_locations).long()  # [B]

        emb = self.location_emb(idx)  # [B, 64]
        out = self.mlp(emb)           # [B, D]

        # Zero out embedding where location was missing
        missing_mask = (location < 0).unsqueeze(1).float()  # [B, 1]
        return out * (1 - missing_mask)
