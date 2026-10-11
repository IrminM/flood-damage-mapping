"""Abstract data-source interfaces.

Prototype implementations read local files (satellite_local, dem_local, osm_local).
Later implementations (e.g. a Copernicus Data Space provider that downloads scenes
for any AOI/date the judges choose) implement the same interfaces, so the
orchestrator does not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from ..core.contracts import AOI, SceneMetadata, SceneRef
from ..core.georaster import GeoRaster


@dataclass
class LoadedScene:
    """A scene on disk (ref) plus its pixels (raster). Band names come from the metadata."""
    ref: SceneRef
    raster: GeoRaster
    warnings: list[str] = field(default_factory=list)


@dataclass
class OSMSnapshotMeta:
    snapshot_date: date
    source: str
    bbox: Optional[tuple[float, float, float, float]] = None
    fetched_at: Optional[str] = None
    filters: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OSMData:
    """Pre-event OSM layers as GeoJSON FeatureCollections (EPSG:4326).

    Conversion to shapely/GeoDataFrames happens in the GIS stage, so this stays a
    plain, serialisable container. Every feature has a stable string `id`.
    """
    meta: OSMSnapshotMeta
    layers: dict[str, dict[str, Any]]
    source_dir: Optional[Path] = None
    missing_layers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def count(self, layer: str) -> int:
        return len(self.layers.get(layer, {}).get("features", []))


class SceneProvider(ABC):
    @abstractmethod
    def load_scene(self, path: Path | str) -> LoadedScene:
        """Load one scene (pixels + validated metadata)."""

    def search(self, aoi: AOI, start: date, end: date, sensor: str) -> list[SceneMetadata]:
        """Find available scenes. Local providers cannot search; remote providers override."""
        raise NotImplementedError(f"{type(self).__name__} does not support scene search")


class DEMProvider(ABC):
    @abstractmethod
    def load_dem(self, path: Path | str) -> GeoRaster:
        """Load an elevation raster (single band named 'elevation', metres)."""


class OSMProvider(ABC):
    @abstractmethod
    def load(self, source: Path | str, event_date: date) -> OSMData:
        """Load a pre-event OSM snapshot; must enforce snapshot_date < event_date."""
