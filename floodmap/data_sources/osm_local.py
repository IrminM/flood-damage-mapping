"""Load a pre-event OpenStreetMap snapshot from a local directory.

Expected layout (written by scripts/fetch_osm_ohsome.py):

    <osm_dir>/
        snapshot_meta.json      {"snapshot_date": "2026-07-27", "source": "ohsome", ...}
        buildings.geojson
        roads.geojson
        bridges.geojson
        places.geojson
        health.geojson

Missing layer files are allowed (e.g. no hospital in a remote AOI) and reported in
`missing_layers`; downstream modules decide whether a missing layer is fatal.
The snapshot date is checked against the event date by the data-rules guard.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from ..core.config import OSMConfig
from ..core.contracts import SourceType
from ..core.data_rules import DataRulesGuard, to_date
from ..core.errors import InputDataError
from .base import OSMData, OSMProvider, OSMSnapshotMeta

SNAPSHOT_META = "snapshot_meta.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise InputDataError(f"Cannot read JSON {path}: {e}") from e


def _feature_id(layer: str, feat: dict[str, Any], index: int) -> str:
    props = feat.get("properties") or {}
    for key in ("@osmId", "osm_id", "id"):
        v = props.get(key) if key != "id" else feat.get("id", props.get("id"))
        if v not in (None, ""):
            return str(v)
    return f"{layer}/{index}"


class LocalOSMProvider(OSMProvider):
    def __init__(self, guard: DataRulesGuard, cfg: OSMConfig):
        self.guard = guard
        self.cfg = cfg

    def load(self, source: Path | str, event_date: date) -> OSMData:
        osm_dir = Path(source)
        if not osm_dir.is_dir():
            raise InputDataError(f"OSM directory not found: {osm_dir}")

        meta_path = osm_dir / SNAPSHOT_META
        if not meta_path.exists():
            raise InputDataError(
                f"{meta_path} missing: the snapshot date must be recorded to prove the OSM data is pre-event"
            )
        raw_meta = _read_json(meta_path)
        if "snapshot_date" not in raw_meta:
            raise InputDataError(f"{meta_path} has no 'snapshot_date'")
        meta = OSMSnapshotMeta(
            snapshot_date=to_date(raw_meta["snapshot_date"]),
            source=str(raw_meta.get("source", "unknown")),
            bbox=tuple(raw_meta["bbox"]) if raw_meta.get("bbox") else None,
            fetched_at=raw_meta.get("fetched_at"),
            filters=raw_meta.get("filters", {}),
            extra={k: v for k, v in raw_meta.items()
                   if k not in {"snapshot_date", "source", "bbox", "fetched_at", "filters"}},
        )
        source_type = raw_meta.get("source_type", SourceType.OSM_PRE_EVENT.value)
        self.guard.check_input(osm_dir, source_type, extra_text=[meta.source])
        self.guard.check_osm_snapshot(meta.snapshot_date, event_date)

        layers: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        warnings: list[str] = []
        for layer in self.cfg.layers:
            path = osm_dir / f"{layer}.geojson"
            if not path.exists():
                missing.append(layer)
                layers[layer] = {"type": "FeatureCollection", "features": []}
                continue
            fc = _read_json(path)
            if fc.get("type") != "FeatureCollection":
                raise InputDataError(f"{path} is not a GeoJSON FeatureCollection")
            layers[layer] = self._validate_layer(layer, fc, warnings)

        return OSMData(meta=meta, layers=layers, source_dir=osm_dir,
                       missing_layers=missing, warnings=warnings)

    def _validate_layer(self, layer: str, fc: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        allowed = set(self.cfg.geometry_types.get(layer, []))
        kept: list[dict[str, Any]] = []
        dropped_type, dropped_empty = 0, 0
        seen: set[str] = set()
        for i, feat in enumerate(fc.get("features", [])):
            geom = feat.get("geometry")
            if not geom or not geom.get("coordinates"):
                dropped_empty += 1
                continue
            if allowed and geom.get("type") not in allowed:
                dropped_type += 1
                continue
            fid = _feature_id(layer, feat, i)
            if fid in seen:  # same OSM element twice (e.g. split by ohsome) -> make unique
                fid = f"{fid}#{i}"
            seen.add(fid)
            kept.append({"type": "Feature", "id": fid, "geometry": geom,
                         "properties": dict(feat.get("properties") or {})})
        if dropped_type:
            warnings.append(f"{layer}: dropped {dropped_type} features with unexpected geometry type "
                            f"(allowed: {sorted(allowed)})")
        if dropped_empty:
            warnings.append(f"{layer}: dropped {dropped_empty} features with empty geometry")
        return {"type": "FeatureCollection", "features": kept}
