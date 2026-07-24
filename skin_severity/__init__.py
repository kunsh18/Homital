"""
Skin Disease Severity Scoring — Gated Neural Architecture
==========================================================

A modular PyTorch architecture that:
  1. Detects / segments the lesion ROI,
  2. Extracts multi-scale features via a shared CNN backbone,
  3. Routes features through 8 specialized gates, each designed
     to learn a distinct family of dermatological severity cues,
  4. Fuses gate embeddings with learned attention weights,
  5. Predicts a final severity score (classification or regression).

Gate specialization is encouraged *architecturally* — through
controlled inputs, different convolution types, attention mechanisms,
and feature-map routing — NOT through separate auxiliary losses.

The model is trained with a single final severity loss.
"""

from .model import SkinSeverityModel
from .roi_extractor import ROIExtractor
from .backbone import SharedBackbone
from .fusion import GatedFusion

__all__ = [
    "SkinSeverityModel",
    "ROIExtractor",
    "SharedBackbone",
    "GatedFusion",
]
