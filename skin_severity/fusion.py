"""
GatedFusion — Attention-Based Gate Fusion (Continuous Severity)
================================================================

Dynamically weights and fuses all gate embeddings into a single
severity embedding.  The attention weights are sample-dependent:
for one image the DamageGate may dominate, for another the
TextureGate may dominate.

Severity is predicted as a CONTINUOUS score in [0.0, 3.0]:
  0.0 – 0.5  → very mild
  0.5 – 1.2  → mild
  1.2 – 2.0  → moderate
  2.0 – 2.6  → severe
  2.6 – 3.0  → critical / extreme

These interpretation bands are NOT used during training — the
model is trained purely with regression loss on continuous labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Severity interpretation bands (for inference only) ──
SEVERITY_BANDS = [
    (0.0, 0.5, "very_mild"),
    (0.5, 1.2, "mild"),
    (1.2, 2.0, "moderate"),
    (2.0, 2.6, "severe"),
    (2.6, 3.0, "critical"),
]

# Maximum severity score (scale factor)
MAX_SEVERITY = 3.0


def interpret_severity(score: float) -> str:
    """Map a continuous severity score to a human-readable band.

    This is for display / logging only — never used in training.
    """
    for low, high, label in SEVERITY_BANDS:
        if low <= score < high:
            return label
    return "critical"  # score == 3.0


class GatedFusion(nn.Module):
    """Attention-weighted fusion → continuous severity regression.

    Parameters
    ----------
    num_gates : int
        Number of gates being fused.
    embed_dim : int
        Dimensionality of each gate embedding (D).
    """

    def __init__(
        self,
        num_gates: int = 8,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.num_gates = num_gates
        self.embed_dim = embed_dim

        # ── Attention network ──
        # Maps the concatenated gate embeddings to per-gate weights.
        self.attn_net = nn.Sequential(
            nn.Linear(num_gates * embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_gates),
            # Softmax applied in forward so raw logits are available
        )

        # ── Post-fusion MLP ──
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # ── Continuous severity regression head ──
        # sigmoid(raw) * MAX_SEVERITY → score ∈ [0.0, 3.0]
        self.severity_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        gate_embeddings: list[torch.Tensor],
        gate_scores: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            gate_embeddings : list of [B, D] tensors (one per gate)
            gate_scores     : list of [B, 1] tensors (internal scores)

        Returns:
            dict with keys:
              'severity'     : [B, 1]  continuous score in [0.0, 3.0]
              'gate_weights' : [B, num_gates]  attention weights
              'fused'        : [B, D]  fused embedding
        """
        # Stack embeddings → [B, G, D]
        stacked = torch.stack(gate_embeddings, dim=1)  # [B, G, D]
        B, G, D = stacked.shape

        # Concatenate for attention input
        flat = stacked.view(B, G * D)  # [B, G*D]

        # Compute attention weights (with NaN protection)
        attn_logits = self.attn_net(flat)  # [B, G]
        attn_logits = attn_logits.clamp(-50, 50)  # prevent extreme values
        attn_logits = torch.nan_to_num(attn_logits, nan=0.0)
        gate_weights = F.softmax(attn_logits, dim=1)  # [B, G]
        # Fallback: if any sample still has NaN weights, use uniform
        nan_mask = gate_weights.isnan().any(dim=1, keepdim=True)  # [B, 1]
        if nan_mask.any():
            uniform = torch.full_like(gate_weights, 1.0 / G)
            gate_weights = torch.where(nan_mask, uniform, gate_weights)

        # Weighted sum of gate embeddings
        # [B, G, 1] * [B, G, D] → sum → [B, D]
        fused = (gate_weights.unsqueeze(2) * stacked).sum(dim=1)  # [B, D]

        # Post-fusion refinement
        fused = self.fusion_mlp(fused)  # [B, D]

        # ── Continuous severity prediction ──
        # sigmoid maps to (0, 1), then scale to (0, 3.0)
        raw = self.severity_head(fused)                   # [B, 1]
        severity = torch.sigmoid(raw) * MAX_SEVERITY      # [B, 1] ∈ [0, 3.0]

        return {
            "severity": severity,
            "gate_weights": gate_weights,
            "fused": fused,
        }
