"""Integration tests for Stage 6 downstream path pipeline."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from floodmap.core.config import Settings
from floodmap.core.contracts import DataMode, DownstreamPathSummary, SourceType
from floodmap.core.errors import DataRuleViolation
from floodmap.core.georaster import GeoRaster
from floodmap.data_sources.base import OSMData, OSMSnapshotMeta
from floodmap.pipeline.stage6_pipeline import run_stage6_pipeline


def _make_mock_osm_data(snapshot_date: date) -> OSMData:
    meta = OSMSnapshotMeta(
        snapshot_date=snapshot_date,
        source="ohsome",
    )
    places_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "node/10",
                "properties": {"osm_id": "node/10", "name": "Valley Village", "place": "village"},
                "geometry": {"type": "Point", "coordinates": [85.02, 27.98]},
            }
        ],
    }
    layers = {
        "places": places_fc,
        "roads": {"type": "FeatureCollection", "features": []},
        "buildings": {"type": "FeatureCollection", "features": []},
        "bridges": {"type": "FeatureCollection", "features": []},
        "health": {"type": "FeatureCollection", "features": []},
    }
    return OSMData(meta=meta, layers=layers)


def _make_mock_dem() -> GeoRaster:
    r_idx, c_idx = np.mgrid[0:40, 0:40]
    elev = 1500.0 - (r_idx * 15.0 + c_idx * 8.0)
    transform = (0.001, 0.0, 85.0, 0.0, -0.001, 28.0)
    data = elev.astype(np.float32)[np.newaxis, ...]
    return GeoRaster(
        data=data,
        transform=transform,
        crs="EPSG:4326",
        nodata=-9999.0,
        metadata={"source_type": SourceType.COPERNICUS_DEM.value},
    )


def test_stage6_pipeline_end_to_end(tmp_path: Path, settings: Settings):
    event_date = date(2026, 8, 26)
    snapshot_date = date(2026, 7, 27)

    dem = _make_mock_dem()
    osm_data = _make_mock_osm_data(snapshot_date)
    run_dir = tmp_path / "run_stage6"

    # Start point near top-left: c=5, r=5 (lon=85.005, lat=27.995)
    upstream = (85.005, 27.995)

    res = run_stage6_pipeline(
        dem_source=dem,
        upstream_point=upstream,
        osm_source=osm_data,
        settings=settings,
        run_dir=run_dir,
        event_date=event_date,
        data_mode=DataMode.SYNTHETIC_TEST,
    )

    # 1. Output files exist
    assert res.outputs["downstream_path"].exists()
    assert res.outputs["downstream_path_settlements"].exists()
    assert res.outputs["summary"].exists()
    assert res.outputs["provenance"].exists()

    # 2. Check summary schema and values
    summary = res.summary
    assert isinstance(summary, DownstreamPathSummary)
    assert summary.start_elevation_m > summary.end_elevation_m
    assert summary.total_elevation_drop_m > 0.0
    assert summary.path_length_km > 0.0
    assert summary.path_vertices_count > 5
    assert summary.upstream_input_coords == (85.005, 27.995)

    # 3. Check GeoJSON
    path_data = json.loads(res.outputs["downstream_path"].read_text())
    assert len(path_data["features"]) == 1
    assert path_data["features"][0]["geometry"]["type"] == "LineString"

    # 4. Check ProvenanceLog
    prov_data = json.loads(res.outputs["provenance"].read_text())
    assert prov_data["data_mode"] == DataMode.SYNTHETIC_TEST.value
    step_names = [s["name"] for s in prov_data["steps"]]
    assert "trace_downstream_path" in step_names
    assert "trace_nearby_settlements" in step_names
    assert "write_downstream_outputs" in step_names


def test_stage6_pipeline_data_rules_guard_post_event(tmp_path: Path, settings: Settings):
    event_date = date(2026, 8, 26)
    post_event_date = date(2026, 8, 28)  # Post-event OSM edit!

    dem = _make_mock_dem()
    osm_data = _make_mock_osm_data(post_event_date)

    with pytest.raises(DataRuleViolation, match="Post-event OSM edits are prohibited"):
        run_stage6_pipeline(
            dem_source=dem,
            upstream_point=(85.005, 27.995),
            osm_source=osm_data,
            settings=settings,
            run_dir=tmp_path / "post_event_run",
            event_date=event_date,
        )


def test_stage6_pipeline_real_dem_smoke(tmp_path: Path, settings: Settings):
    real_dem_path = Path("data/dem/dev/dem_glo30.tif")
    if not real_dem_path.exists():
        pytest.skip("Real Copernicus DEM dem_glo30.tif not present")

    event_date = date(2026, 8, 26)
    snapshot_date = date(2026, 7, 27)
    osm_data = _make_mock_osm_data(snapshot_date)

    # Real point near Trishuli upper catchment
    upstream = (85.25, 28.15)
    run_dir = tmp_path / "smoke_real_dem"

    res = run_stage6_pipeline(
        dem_source=real_dem_path,
        upstream_point=upstream,
        osm_source=osm_data,
        settings=settings,
        run_dir=run_dir,
        event_date=event_date,
        data_mode=DataMode.REAL,
    )

    assert res.summary.path_length_km > 0.0
    assert res.summary.start_elevation_m > 0.0
    assert res.outputs["downstream_path"].exists()
