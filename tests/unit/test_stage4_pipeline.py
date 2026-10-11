"""Integration tests for Stage 4 pipeline (floodmap.pipeline.stage4_pipeline)."""
import json
from datetime import date

import numpy as np
import pytest

from floodmap.core.config import load_settings
from floodmap.core.contracts import DataMode, InfrastructureSummary, OutputClass
from floodmap.core.errors import DataRuleViolation
from floodmap.core.georaster import GeoRaster
from floodmap.pipeline.stage4_pipeline import run_stage4_pipeline

EVENT_DATE = date(2026, 8, 26)


@pytest.fixture
def synthetic_stage4_raster():
    # 100x100 raster at (85.0, 27.9) in EPSG:32645 equivalent or projected coordinates
    # For test, we place coordinates matching osm_dir_factory bbox [85.0, 27.9, 85.1, 28.0]
    # Let's use EPSG:4326 or EPSG:32645. Since GeoRaster supports any CRS:
    # We use EPSG:4326 directly so geometries match osm_dir_factory coordinates:
    # osm_dir_factory has:
    # building at [[[85.01, 27.91], [85.011, 27.91], [85.011, 27.911], [85.01, 27.911], [85.01, 27.91]]]
    # road from [85.0, 27.9] to [85.05, 27.95]
    H, W = 100, 100
    # res = 0.001 deg (~100m)
    transform = (0.001, 0.0, 85.0, 0.0, -0.001, 28.0)
    data = np.zeros((1, H, W), dtype=np.uint8)
    # Put flood water over building at (85.01, 27.91):
    # col: (85.010 - 85.0) / 0.001 = 10
    # row: (28.0 - 27.910) / 0.001 = 90
    data[0, 85:95, 5:15] = int(OutputClass.FLOOD_WATER)

    return GeoRaster(data=data, transform=transform, crs="EPSG:4326", nodata=255)


def test_stage4_pipeline_end_to_end(tmp_path, synthetic_stage4_raster, osm_dir_factory, settings):
    osm_dir = osm_dir_factory(tmp_path / "osm", snapshot_date="2026-07-27")
    out_dir = tmp_path / "stage4_run"

    res = run_stage4_pipeline(
        classes_raster=synthetic_stage4_raster,
        osm_source=osm_dir,
        settings=settings,
        run_dir=out_dir,
        event_date=EVENT_DATE,
        analysis_id="s4_test",
        data_mode=DataMode.SYNTHETIC_TEST,
    )

    assert (out_dir / "infrastructure_summary.json").exists()
    assert (out_dir / "affected_buildings.geojson").exists()
    assert (out_dir / "affected_roads.geojson").exists()
    assert (out_dir / "affected_bridges.geojson").exists()
    assert (out_dir / "flood_evidence_vector.geojson").exists()
    assert (out_dir / "provenance.json").exists()

    # Validate Summary contract
    summary = InfrastructureSummary.model_validate_json((out_dir / "infrastructure_summary.json").read_text(encoding="utf-8"))
    assert summary.data_mode is DataMode.SYNTHETIC_TEST
    assert summary.osm_snapshot_date == date(2026, 7, 27)
    assert summary.buildings.total_count == 1
    assert summary.buildings.potentially_affected_count == 1
    assert summary.roads.total_segments == 1
    assert summary.roads.potentially_affected_segments == 1

    # Validate Provenance
    prov_data = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov_data["analysis_id"] == "s4_test"
    steps = [s["name"] for s in prov_data.get("steps", [])]
    assert "vectorize_evidence" in steps
    assert "assess_buildings" in steps
    assert "assess_roads" in steps


def test_stage4_pipeline_rejects_post_event_osm(tmp_path, synthetic_stage4_raster, osm_dir_factory, settings):
    # Post-event snapshot must be rejected by DataRulesGuard
    osm_dir = osm_dir_factory(tmp_path / "osm_post", snapshot_date="2026-08-30")
    out_dir = tmp_path / "stage4_reject"

    with pytest.raises(DataRuleViolation, match="Post-event OSM edits are prohibited"):
        run_stage4_pipeline(
            classes_raster=synthetic_stage4_raster,
            osm_source=osm_dir,
            settings=settings,
            run_dir=out_dir,
            event_date=EVENT_DATE,
        )
