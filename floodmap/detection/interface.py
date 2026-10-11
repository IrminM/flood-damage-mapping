"""Change-detection interface (traditional remote sensing, path A).

This path is deliberately separate from AI segmentation (path B, stage 3). Both
produce rasters on the same analysis grid using the canonical class codes; a later
fusion step combines them and records per-pixel evidence (change / AI / both).

To add a better method: implement `ChangeDetector`, register it in `REGISTRY`, and
select it with `change_detection.method` in config. Nothing downstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from ..core.config import Settings
from ..core.contracts import OutputClass
from ..core.georaster import GeoRaster
from ..satellite.preprocess import PreprocessedPair
from ..satellite.terrain import TerrainProducts

# Human-readable meaning of each class *as produced by change detection*.
# Class 3 is NOT "debris": it is any significant non-water backscatter change.
CHANGE_CLASS_LABELS: dict[int, str] = {
    int(OutputClass.BACKGROUND): "no detected change",
    int(OutputClass.FLOOD_WATER): "new open water (dark after, darkened by >= threshold)",
    int(OutputClass.PERMANENT_WATER): "water in both dates (river/lake)",
    int(OutputClass.DEBRIS_OR_CHANGE): "significant non-water backscatter change "
                                       "(possible debris/damage/landslide/vegetation change; unverified)",
    int(OutputClass.NODATA): "not observable (outside AOI, no data, radar shadow/layover)",
}


@dataclass
class ChangeResult:
    classes: GeoRaster              # uint8, canonical OutputClass codes, nodata 255
    log_ratio_db: GeoRaster         # float32 bands e.g. ["VV_log_ratio", "VH_log_ratio"] (after - before)
    exclusion: np.ndarray           # uint8 ExclusionFlag bits
    method: str
    params: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    context: Optional[np.ndarray] = None   # uint8 ChangeContext labels for class-3 pixels (optional)


class ChangeDetector(Protocol):
    name: str

    def detect(self, pair: PreprocessedPair, terrain: Optional[TerrainProducts]) -> ChangeResult: ...


REGISTRY: dict[str, Callable[[Settings], ChangeDetector]] = {}


def register(name: str):
    def deco(factory: Callable[[Settings], ChangeDetector]):
        REGISTRY[name] = factory
        return factory
    return deco


def get_change_detector(settings: Settings) -> ChangeDetector:
    from . import change_s1  # noqa: F401  (registers built-in methods)

    name = settings.change_detection.method
    if name not in REGISTRY:
        raise ValueError(f"Unknown change_detection.method {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](settings)
