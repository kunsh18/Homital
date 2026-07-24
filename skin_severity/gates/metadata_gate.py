"""
MetadataGate — Patient-Reported Symptom Features
==================================================

Purpose:
  Incorporate patient-reported severity-related symptoms such as
  pain, itching, burning, fever, sleep disturbance, duration.

Design choices:
  • Consumes a generic metadata tensor [B, M].
  • Small MLP with layer normalisation and dropout.
  • Returns a zero embedding when metadata is missing (all zeros or
    not provided).
"""

import torch
import torch.nn as nn

from .base_gate import FeatureGate


class MetadataGate(FeatureGate):
    """Specialised gate for patient-reported metadata.

    Inputs consumed:
      metadata : [B, M]  (float tensor, may be all zeros if missing)
    """

    def __init__(
        self,
        metadata_dim: int = 10,
        embed_dim: int = 128,
    ):
        super().__init__(embed_dim=embed_dim)
        self.metadata_dim = metadata_dim

        self.mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def _gate_forward(self, **kwargs) -> torch.Tensor:
        metadata: torch.Tensor | None = kwargs.get("metadata", None)

        if metadata is None:
            B = kwargs.get("_batch_size", 1)
            device = next(self.parameters()).device
            return torch.zeros(B, self.embed_dim, device=device)

        out = self.mlp(metadata)  # [B, D]

        # Zero out when all metadata values are zero (missing)
        has_data = (metadata.abs().sum(dim=1, keepdim=True) > 0).float()  # [B, 1]
        return out * has_data
