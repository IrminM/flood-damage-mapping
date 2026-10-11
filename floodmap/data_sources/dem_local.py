"""Load a local Copernicus DEM (GLO-30 / WorldDEM-30) GeoTIFF.

An optional sidecar `<dem>.meta.json` may declare {"source_type": ...}; it is written
by scripts/fetch_dem_glo30.py (copernicus-dem) and by dev_mocks (synthetic-test).
Without a sidecar the DEM is assumed to be Copernicus DEM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.contracts import SourceType
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError
from ..core.georaster import GeoRaster
from ..io.geotiff import read_geotiff
from .base import DEMProvider

# Plausible elevation range on Earth (m); values outside are treated as invalid.
_MIN_ELEV, _MAX_ELEV = -500.0, 9000.0


def read_dem_sidecar(path: Path) -> Optional[dict[str, Any]]:
    for c in (path.with_suffix(".meta.json"), path.with_name(path.name + ".meta.json")):
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception as e:
                raise InputDataError(f"Invalid DEM sidecar {c}: {e}") from e
    return None


class LocalDEMProvider(DEMProvider):
    def __init__(self, guard: DataRulesGuard, source_type: Optional[SourceType] = None):
        self.guard = guard
        self.source_type = source_type    # None -> sidecar, else copernicus-dem

    def load_dem(self, path: Path | str) -> GeoRaster:
        path = Path(path)
        side = read_dem_sidecar(path) or {}
        source = self.source_type or SourceType(side.get("source_type", SourceType.COPERNICUS_DEM.value))
        if source not in (SourceType.COPERNICUS_DEM, SourceType.SYNTHETIC_TEST):
            raise InputDataError(f"DEM {path}: source_type {source.value} is not a DEM source")
        self.guard.check_input(path, source, extra_text=[str(side.get("product", ""))])
        dem = read_geotiff(path, bands=[1])
        arr = dem.data.astype(np.float32)
        invalid = ~dem.valid_mask() | (arr[0] < _MIN_ELEV) | (arr[0] > _MAX_ELEV)
        arr[0][invalid] = np.nan
        out = dem.with_data(arr, band_names=["elevation"], nodata=np.nan)
        if not np.isfinite(out.data).any():
            raise InputDataError(f"DEM {path} has no valid elevation values")
        out.metadata.update({"source_type": source.value, "units": "m", "sidecar": side})
        return out
