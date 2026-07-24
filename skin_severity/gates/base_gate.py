"""
FeatureGate — Abstract base class for all specialised gates.
=============================================================

Every gate must return:
  embedding : [B, D]    — the gate's feature embedding
  score     : [B, 1]    — an internal activation / confidence score

Sub-classes implement ``_gate_forward`` which receives whichever
inputs that particular gate needs (feature maps, ROI image, mask,
metadata, etc.).  The base class standardises output projection
and the activation score head.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FeatureGate(ABC, nn.Module):
    """Abstract base for all dermatological feature gates.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the gate embedding (D).
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Shared output projection (sub-classes may override) ──
        # Projects the gate's raw representation to the unified
        # embedding space so that GatedFusion can compare / combine
        # embeddings from heterogeneous gates.
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ── Internal activation score ──
        # A single scalar per sample indicating how "active" or
        # "confident" this gate is for the given input.
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    # ── abstract interface ──

    @abstractmethod
    def _gate_forward(self, **kwargs) -> torch.Tensor:
        """Compute a raw embedding [B, D] from gate-specific inputs.

        Sub-classes decide which kwargs they consume (e.g. ``low``,
        ``mid``, ``high``, ``roi_image``, ``mask``, ``box``,
        ``metadata``).
        """
        ...

    # ── public forward ──

    def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        embedding : [B, D]
        score     : [B, 1]
        """
        raw = self._gate_forward(**kwargs)       # [B, D]
        raw = torch.nan_to_num(raw, nan=0.0)     # safety: replace any NaN
        embedding = self.out_proj(raw)            # [B, D]
        embedding = torch.nan_to_num(embedding, nan=0.0)
        score = self.score_head(raw.detach())     # [B, 1] — detached so
        #   the score head does not create an auxiliary gradient path
        return embedding, score
