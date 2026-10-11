"""AI flood segmentation interface and contracts (Stage 3, Path B).

Defines the AISegmentor protocol and AISegmentationResult container.
Any AI segmentation model (the baseline S1-MiniUNet-v1 or future plug-in models)
must implement this protocol to be consumed by the fusion layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from ..core.georaster import GeoRaster
from ..satellite.preprocess import PreprocessedPair
from ..satellite.terrain import TerrainProducts


@dataclass
class AISegmentationResult:
    """Standardized output from an AI flood segmentation model."""
    probability: GeoRaster         # float32 [0.0, 1.0], 1 band 'water_prob', NaN nodata
    water_mask: np.ndarray         # bool (H, W): True where flood water is detected
    model_name: str                # Identifier of the model
    threshold: float               # Decision threshold used (initial/default until calibrated)
    metadata: dict = field(default_factory=dict)   # Preprocessing params, tile size, weights hash
    stats: dict = field(default_factory=dict)      # Pixel counts, mean probability, estimated area


class AISegmentor(Protocol):
    """Protocol that all AI segmentation implementations must satisfy."""
    name: str

    def segment(
        self,
        pair: PreprocessedPair,
        terrain: Optional[TerrainProducts] = None,
        s2_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> AISegmentationResult:
        """Run AI segmentation on a preprocessed Sentinel-1 pair (with optional Sentinel-2 input)."""
        ...



AI_REGISTRY: dict[str, Callable[[Any], AISegmentor]] = {}


def register_ai(name: str):
    def deco(factory: Callable[[Any], AISegmentor]):
        AI_REGISTRY[name] = factory
        return factory
    return deco


def get_ai_segmentor(settings: Any) -> AISegmentor:
    from . import segmentor  # noqa: F401 (registers built-in methods)
    from .adapters import model_a_adapter  # noqa: F401 (registers model_a)
    from .adapters import attention_unet_adapter  # noqa: F401 (registers attention_unet_s1_s2)

    name = settings.ai.model
    if name not in AI_REGISTRY:
        raise ValueError(f"Unknown ai.model {name!r}; available: {sorted(AI_REGISTRY)}")
    return AI_REGISTRY[name](settings)
