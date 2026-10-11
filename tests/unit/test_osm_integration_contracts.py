"""Unit and integration tests for Pre-Event OSM data readiness, availability detection,
missing-data guarantees, and Stage 4/5 integration contracts.
"""
from datetime import date
import json
from pathlib import Path
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from dev_mocks.synthetic_scene import synthetic_aoi, write_synthetic_pair
from floodmap.core.contracts import (
    AnalysisSubmitRequest,
    ConnectivitySummary,
    DataAvailabilityStatus,
    DataMode,
    InfrastructureSummary,
    OutputClass,
)
from floodmap.core.errors import DataRuleViolation
from floodmap.core.georaster import GeoRaster
from floodmap.data_sources.base import OSMData, OSMSnapshotMeta
from floodmap.data_sources.ohsome import OhsomeRequest, fetch_layer
from floodmap.data_sources.osm_local import LocalOSMProvider
from floodmap.pipeline.orchestrator import (
    _check_osm_availability,
    generate_situation_report,
    run_full_pipeline,
)
from floodmap.pipeline.stage4_pipeline import run_stage4_pipeline
from floodmap.pipeline.stage5_pipeline import run_stage5_pipeline

EVENT_DATE = date(2026, 8, 26)
VALID_PRE_EVENT_SNAPSHOT = date(2026, 7, 27)
POST_EVENT_SNAPSHOT = date(2026, 8, 30)


# ============================================================================
# 1. OSM Availability Detection Tests
# ============================================================================

def test_check_osm_availability_none_or_missing_path(tmp_path):
    """None or nonexistent path must return False with explanatory reason."""
    ok, reason = _check_osm_availability(None)
    assert not ok
    assert "not provided" in reason

    ok, reason = _check_osm_availability(tmp_path / "does_not_exist")
    assert not ok
    assert "does not exist" in reason


def test_check_osm_availability_missing_metadata(tmp_path):
    """Directory with layers but no snapshot_meta.json must be flagged as unavailable."""
    osm_dir = tmp_path / "osm_no_meta"
    osm_dir.mkdir()
    (osm_dir / "roads.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    (osm_dir / "buildings.geojson").write_text('{"type": "FeatureCollection", "features": []}')

    ok, reason = _check_osm_availability(osm_dir)
    assert not ok
    assert "snapshot_meta.json" in reason


def test_check_osm_availability_missing_layers(tmp_path):
    """Directory with metadata but missing required layer files must be flagged as unavailable."""
    osm_dir = tmp_path / "osm_partial"
    osm_dir.mkdir()
    (osm_dir / "snapshot_meta.json").write_text('{"snapshot_date": "2026-07-27"}')
    (osm_dir / "roads.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    # buildings.geojson missing

    ok, reason = _check_osm_availability(osm_dir)
    assert not ok
    assert "missing pre-event layers" in reason


def test_check_osm_availability_valid(tmp_path, osm_dir_factory):
    """Directory with snapshot_meta.json and required layers must return True."""
    osm_dir = osm_dir_factory(tmp_path / "osm_valid", snapshot_date="2026-07-27")
    ok, reason = _check_osm_availability(osm_dir)
    assert ok
    assert reason is None


# ============================================================================
# 2. Correct Pre-Event Loading & Data-Rule Enforcement
# ============================================================================

def test_correct_preevent_osm_loading(tmp_path, guard, settings, osm_dir_factory):
    """Pre-event OSM data loads properly with correct dates and tags."""
    osm_dir = osm_dir_factory(tmp_path / "osm_preevent", snapshot_date="2026-07-27")
    provider = LocalOSMProvider(guard, settings.osm)
    data = provider.load(osm_dir, EVENT_DATE)

    assert data.meta.snapshot_date == VALID_PRE_EVENT_SNAPSHOT
    assert data.count("buildings") >= 1
    assert data.count("roads") >= 1
    assert data.layers["roads"]["features"][0]["id"] == "way/1"


def test_preevent_osm_loader_rejects_postevent(tmp_path, guard, settings, osm_dir_factory):
    """Post-event OSM snapshot must be rejected with DataRuleViolation."""
    osm_dir = osm_dir_factory(tmp_path / "osm_post", snapshot_date=POST_EVENT_SNAPSHOT.isoformat())
    provider = LocalOSMProvider(guard, settings.osm)

    with pytest.raises(DataRuleViolation, match="Post-event OSM edits are prohibited"):
        provider.load(osm_dir, EVENT_DATE)


# ============================================================================
# 3. Missing Data Behavior & No Fake Zero-Impact Figures
# ============================================================================

def test_missing_osm_never_becomes_fake_zero_impact(tmp_path, settings):
    """When OSM data is absent, the orchestrator and sitrep MUST NOT claim 0 buildings affected."""
    bpath, apath, pair = write_synthetic_pair(tmp_path / "sar")
    aoi = synthetic_aoi(pair, inset_m=30)
    dem_path = tmp_path / "sar" / "SYNTHETIC_dem.tif"

    req = AnalysisSubmitRequest(
        aoi=aoi.geometry,
        event_date=EVENT_DATE,
        before_scene=str(bpath),
        after_scene=str(apath),
        dem=str(dem_path),
        osm_dir=str(tmp_path / "missing_osm_folder"),
        name="No OSM Test",
        sync=True,
    )

    result = run_full_pipeline(req, settings=settings, base_runs_dir=tmp_path / "runs")

    # Pipeline status
    assert result.data_availability.osm_preevent == DataAvailabilityStatus.UNAVAILABLE
    assert result.stages_executed["stage4_infrastructure"] == "UNAVAILABLE_NO_PREEVENT_OSM"
    assert result.stages_executed["stage5_connectivity"] == "UNAVAILABLE_NO_ROAD_NETWORK"
    assert result.stage4_summary is None
    assert result.stage5_summary is None

    # Situation report guarantees
    sitrep = generate_situation_report(result)
    assert sitrep.infrastructure_findings["status"] == "UNAVAILABLE"
    assert "not available" in sitrep.infrastructure_findings["message"].lower()
    assert "buildings_potentially_affected" not in sitrep.infrastructure_findings

    assert sitrep.connectivity_findings["status"] == "UNAVAILABLE"
    assert "unavailable" in sitrep.connectivity_findings["message"].lower()
    assert "settlements_cut_off" not in sitrep.connectivity_findings


# ============================================================================
# 4. Stage 4 Receiving Buildings, Roads, and Bridges
# ============================================================================

def test_stage4_receives_buildings_roads_bridges(tmp_path, settings):
    """Stage 4 receives buildings, roads, and bridges and evaluates each layer."""
    # Synthetic raster with flood water over coordinates
    transform = (0.001, 0.0, 85.0, 0.0, -0.001, 28.0)
    data = np_zeros = __import__("numpy").zeros((1, 100, 100), dtype=__import__("numpy").uint8)
    data[0, 80:95, 5:25] = int(OutputClass.FLOOD_WATER)
    raster = GeoRaster(data=data, transform=transform, crs="EPSG:4326", nodata=255)

    meta = OSMSnapshotMeta(snapshot_date=VALID_PRE_EVENT_SNAPSHOT, source="ohsome")
    buildings_fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "way/b1",
            "properties": {"@osmId": "way/b1", "building": "yes"},
            "geometry": {"type": "Polygon", "coordinates": [[[85.01, 27.91], [85.015, 27.91], [85.015, 27.915], [85.01, 27.915], [85.01, 27.91]]]},
        }],
    }
    roads_fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "way/r1",
            "properties": {"@osmId": "way/r1", "highway": "secondary"},
            "geometry": {"type": "LineString", "coordinates": [[85.0, 27.9], [85.03, 27.93]]},
        }],
    }
    bridges_fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "way/br1",
            "properties": {"@osmId": "way/br1", "bridge": "yes", "highway": "secondary"},
            "geometry": {"type": "LineString", "coordinates": [[85.01, 27.91], [85.02, 27.92]]},
        }],
    }
    osm_data = OSMData(
        meta=meta,
        layers={
            "buildings": buildings_fc,
            "roads": roads_fc,
            "bridges": bridges_fc,
            "places": {"type": "FeatureCollection", "features": []},
            "health": {"type": "FeatureCollection", "features": []},
        },
    )

    out_dir = tmp_path / "stage4_all_infra"
    res = run_stage4_pipeline(
        classes_raster=raster,
        osm_source=osm_data,
        settings=settings,
        run_dir=out_dir,
        event_date=EVENT_DATE,
        data_mode=DataMode.SYNTHETIC_TEST,
    )

    # All three infrastructure output files exist
    assert res.outputs["affected_buildings"].exists()
    assert res.outputs["affected_roads"].exists()
    assert res.outputs["affected_bridges"].exists()

    summary = res.summary
    assert summary.buildings.total_count == 1
    assert summary.roads.total_segments == 1
    assert summary.bridges.total_count == 1
    assert summary.bridges.potentially_affected_count >= 1


# ============================================================================
# 5. Stage 5 Receiving Road Network and Settlements
# ============================================================================

def test_stage5_receives_road_network(tmp_path, settings):
    """Stage 5 constructs road graph and evaluates cut-off settlements from OSM roads and places."""
    meta = OSMSnapshotMeta(snapshot_date=VALID_PRE_EVENT_SNAPSHOT, source="ohsome")
    roads_fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "way/r10",
            "properties": {"@osmId": "way/r10", "highway": "primary"},
            "geometry": {"type": "LineString", "coordinates": [[85.1, 27.9], [85.2, 27.9]]},
        }],
    }
    places_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "node/dest_town",
                "properties": {"@osmId": "node/dest_town", "name": "Hub Town", "place": "town"},
                "geometry": {"type": "Point", "coordinates": [85.2, 27.9]},
            },
            {
                "type": "Feature",
                "id": "node/village_1",
                "properties": {"@osmId": "node/village_1", "name": "Isolated Village", "place": "village"},
                "geometry": {"type": "Point", "coordinates": [85.1, 27.9]},
            },
        ],
    }
    osm_data = OSMData(
        meta=meta,
        layers={
            "roads": roads_fc,
            "places": places_fc,
            "buildings": {"type": "FeatureCollection", "features": []},
            "bridges": {"type": "FeatureCollection", "features": []},
            "health": {"type": "FeatureCollection", "features": []},
        },
    )

    # Disrupted road segment
    aff_gdf = gpd.GeoDataFrame(
        [{"osm_id": "way/r10", "highway": "primary", "potentially_affected": True, "geometry": LineString([(85.1, 27.9), (85.2, 27.9)])}],
        crs="EPSG:4326",
    )
    aff_path = tmp_path / "aff_roads.geojson"
    aff_gdf.to_file(aff_path, driver="GeoJSON")

    res = run_stage5_pipeline(
        affected_roads=aff_path,
        osm_source=osm_data,
        settings=settings,
        run_dir=tmp_path / "s5_run",
        event_date=EVENT_DATE,
        data_mode=DataMode.SYNTHETIC_TEST,
    )

    assert res.summary.disconnected_count == 1
    assert res.summary.affected_road_segments_count == 1
    assert res.outputs["cut_off_settlements"].exists()


# ============================================================================
# 6. ohsome Client API Key & Error Handling
# ============================================================================

def test_ohsome_fetch_layer_raises_on_403():
    """ohsome fetch_layer must explain HeiGIT API key requirement when 403 Forbidden is returned."""
    class Fake403Resp:
        status_code = 403
        def raise_for_status(self):
            raise Exception("403 Forbidden")

    req = OhsomeRequest(
        layer="roads",
        url="https://api.ohsome.org/v1/elements/geometry",
        params={},
        bbox=(85.0, 27.9, 85.1, 28.0),
    )

    with pytest.raises(RuntimeError, match="HeiGIT ohsome API requires an API key"):
        fetch_layer(req, post=lambda url, data, timeout, headers: Fake403Resp())


def test_ohsome_fetch_layer_sends_authorization_header():
    """ohsome fetch_layer includes Authorization header when api_key is supplied."""
    captured_headers = {}

    class Fake200Resp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"type": "FeatureCollection", "features": []}

    def fake_post(url, data, timeout, headers=None):
        nonlocal captured_headers
        captured_headers = headers or {}
        return Fake200Resp()

    req = OhsomeRequest(
        layer="roads",
        url="https://api.ohsome.org/v1/elements/geometry",
        params={},
        bbox=(85.0, 27.9, 85.1, 28.0),
    )

    fetch_layer(req, post=fake_post, api_key="secret-token-123")
    assert captured_headers.get("Authorization") == "secret-token-123"
    assert "FloodDamageMapper" in captured_headers.get("User-Agent", "")
