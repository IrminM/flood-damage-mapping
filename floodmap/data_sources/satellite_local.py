"""Load satellite scenes from local GeoTIFFs + sidecar metadata.

No filename conventions are assumed. Each scene file `X.tif` must have a sidecar
`X.meta.json` (or `X.tif.meta.json`) that validates as `SceneMetadata`. The fetch
scripts write these sidecars automatically; for hand-supplied files, copy
`docs/examples/scene.meta.example.json` and fill it in.

Why a sidecar: orbit track, pass direction, acquisition time and units cannot be
reliably recovered from an arbitrary GeoTIFF, and change detection is only valid
when they are known (same relative orbit, same direction).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.contracts import SceneMetadata, SceneRef
from ..core.data_rules import DataRulesGuard
from ..core.errors import InputDataError
from ..io.geotiff import read_geotiff
from .base import LoadedScene, SceneProvider


def sidecar_path(path: Path) -> Path:
    """Return the existing sidecar path for `path` (X.meta.json preferred, then X.tif.meta.json)."""
    candidates = [path.with_suffix(".meta.json"), path.with_name(path.name + ".meta.json")]
    for c in candidates:
        if c.exists():
            return c
    raise InputDataError(
        f"No metadata sidecar for {path}. Expected one of: {[str(c) for c in candidates]}"
    )


def read_scene_metadata(path: Path | str) -> SceneMetadata:
    sc = sidecar_path(Path(path))
    try:
        return SceneMetadata.model_validate(json.loads(sc.read_text(encoding="utf-8")))
    except Exception as e:
        raise InputDataError(f"Invalid scene metadata in {sc}: {e}") from e


def write_scene_metadata(path: Path | str, meta: SceneMetadata) -> Path:
    """Write the sidecar next to a scene file (used by fetch scripts and dev_mocks)."""
    sc = Path(path).with_suffix(".meta.json")
    sc.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return sc


def unit_sanity_warnings(meta: SceneMetadata, data: np.ndarray, valid: np.ndarray) -> list[str]:
    """Cheap plausibility checks that catch mislabelled units (linear vs dB)."""
    warnings: list[str] = []
    if meta.sensor != "S1" or not valid.any():
        return warnings
    vals = data[:, valid]
    med = float(np.nanmedian(vals))
    if meta.units == "linear":
        if (vals < 0).mean() > 0.01:
            warnings.append(f"Declared linear backscatter but {100 * (vals < 0).mean():.1f}% of values are negative "
                            "(looks like dB)")
        if med > 5:
            warnings.append(f"Declared linear backscatter but median is {med:.2f} (unusually high)")
    elif meta.units == "dB":
        if med > 10 or med < -40:
            warnings.append(f"Declared dB backscatter but median is {med:.2f} (outside typical -40..10 dB)")
        if (vals >= 0).mean() > 0.95:
            warnings.append("Declared dB backscatter but >95% of values are >= 0 (looks like linear)")
    return warnings


class LocalSceneProvider(SceneProvider):
    def __init__(self, guard: DataRulesGuard):
        self.guard = guard

    def load_scene(self, path: Path | str) -> LoadedScene:
        path = Path(path)
        meta = read_scene_metadata(path)
        ref = SceneRef(path=path, metadata=meta)
        self.guard.check_scene(ref)

        raster = read_geotiff(path)
        if raster.count != len(meta.bands):
            raise InputDataError(
                f"{path}: file has {raster.count} bands but metadata declares {len(meta.bands)} {meta.bands}"
            )
        raster.band_names = list(meta.bands)
        raster.metadata.update({"sensor": meta.sensor, "units": meta.units,
                                "acquisition_time": meta.acquisition_time.isoformat(),
                                "data_mode": meta.data_mode.value})
        valid = raster.valid_mask()
        if not valid.any():
            raise InputDataError(f"{path}: no valid pixels (all nodata/NaN)")
        warnings = unit_sanity_warnings(meta, raster.data, valid)
        return LoadedScene(ref=ref, raster=raster, warnings=warnings)
