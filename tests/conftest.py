"""Shared pytest fixtures."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floodmap.core.config import Settings, load_settings  # noqa: E402
from floodmap.core.contracts import SceneMetadata, SourceType  # noqa: E402
from floodmap.core.data_rules import DataRulesGuard  # noqa: E402

EVENT_DATE = date(2026, 8, 26)


@pytest.fixture
def settings() -> Settings:
    # Empty environ => only config/default.yaml; tests never depend on a developer's .env
    return load_settings(environ={})


@pytest.fixture
def guard(settings: Settings) -> DataRulesGuard:
    return DataRulesGuard(settings.data_rules)


def make_s1_meta(day: str, orbit: int = 85, direction: str = "ascending", platform: str = "sentinel-1a",
                 **kw) -> SceneMetadata:
    base = dict(
        sensor="S1", platform=platform, product_type="GRD",
        acquisition_time=datetime.fromisoformat(day + "T00:15:00+00:00"),
        bands=["VV", "VH"], units="linear", processing="test",
        source_type=SourceType.SENTINEL_1, relative_orbit=orbit, orbit_direction=direction,
    )
    base.update(kw)
    return SceneMetadata(**base)


@pytest.fixture
def s1_meta_factory():
    return make_s1_meta


def write_osm_dir(path: Path, snapshot_date: str = "2026-07-27", layers: dict | None = None) -> Path:
    """Tiny hand-made OSM snapshot directory (unit-test fixture, not real data)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshot_meta.json").write_text(json.dumps({
        "snapshot_date": snapshot_date, "source": "ohsome API (OpenStreetMap history)",
        "source_type": "osm-pre-event", "bbox": [85.0, 27.9, 85.1, 28.0],
    }))
    layers = layers if layers is not None else {
        "buildings": [{"type": "Polygon", "coordinates": [[[85.01, 27.91], [85.011, 27.91], [85.011, 27.911],
                                                           [85.01, 27.911], [85.01, 27.91]]]}],
        "roads": [{"type": "LineString", "coordinates": [[85.0, 27.9], [85.05, 27.95]]},
                  {"type": "Point", "coordinates": [85.0, 27.9]}],  # wrong type -> dropped
        "places": [{"type": "Point", "coordinates": [85.02, 27.92]}],
    }
    for name, geoms in layers.items():
        feats = [{"type": "Feature", "geometry": g, "properties": {"@osmId": f"way/{i + 1}"}}
                 for i, g in enumerate(geoms)]
        (path / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return path


@pytest.fixture
def osm_dir_factory():
    return write_osm_dir
