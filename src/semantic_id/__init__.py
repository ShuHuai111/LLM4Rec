"""Compatibility package for the reorganized Semantic ID modules."""

from src.representations.semantic_id import (
    ItemTextEncoder,
    ResidualVectorQuantizer,
    AdvancedResidualVectorQuantizer,
)
from src.representations.semantic_id.mapper import SemanticIDMapper

__all__ = [
    "ItemTextEncoder",
    "ResidualVectorQuantizer",
    "AdvancedResidualVectorQuantizer",
    "SemanticIDMapper",
]
