"""Specialized feature gates for dermatological severity scoring."""

from .base_gate import FeatureGate
from .color_gate import ColorGate
from .texture_gate import TextureGate
from .shape_border_gate import ShapeBorderGate
from .morphology_gate import MorphologyGate
from .damage_gate import DamageGate
from .area_spread_gate import AreaSpreadGate
from .location_risk_gate import LocationRiskGate
from .metadata_gate import MetadataGate

__all__ = [
    "FeatureGate",
    "ColorGate",
    "TextureGate",
    "ShapeBorderGate",
    "MorphologyGate",
    "DamageGate",
    "AreaSpreadGate",
    "LocationRiskGate",
    "MetadataGate",
]
