"""Integration and governance tests for Stage 5 pipeline."""
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from floodmap.core.config import Settings
from floodmap.core.contracts import ConnectivitySummary, DataMode, SourceType
from floodmap.core.errors import DataRuleViolation
from floodmap.data_sources.base import OSMData, OSMSnapshotMeta
from floodmap.pipeline.stage5_pipeline import run_stage5_pipeline


def _make_mock_osm_data(snapshot_date: date) -> OSMData:
    meta = OSMSnapshotMeta(
        snapshot_date=snapshot_date,
        source="ohsome",
    )
    # Simple line road: (85.1, 27.9) to (85.2, 27.9)
    roads_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "way/101",
                "properties": {"osm_id": "way/101", "highway": "primary"},
                "geometry": {"type": "LineString", "coordinates": [[85.1, 27.9], [85.2, 27.9]]},
            }
        ],
    }
    places_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "node/1",
                "properties": {"osm_id": "node/1", "name": "Test Town", "place": "town"},
                "geometry": {"type": "Point", "coordinates": [85.2, 27.9]},
            },
            {
                "type": "Feature",
                "id": "node/2",
                "properties": {"osm_id": "node/2", "name": "Test Village", "place": "village"},
                "geometry": {"type": "Point", "coordinates": [85.1, 27.9]},
            },
        ],
    }
    layers = {
        "roads": roads_fc,
        "places": places_fc,
        "buildings": {"type": "FeatureCollection", "features": []},
        "bridges": {"type": "FeatureCollection", "features": []},
        "health": {"type": "FeatureCollection", "features": []},
    }
    return OSMData(meta=meta, layers=layers)


def test_stage5_pipeline_end_to_end(tmp_path: Path, settings: Settings):
    event_date = date(2026, 8, 26)
    snapshot_date = date(2026, 7, 27)

    osm_data = _make_mock_osm_data(snapshot_date)

    # Mock affected roads: way/101 is affected
    affected_roads_gdf = gpd.GeoDataFrame(
        [
            {
                "osm_id": "way/101",
                "highway": "primary",
                "potentially_affected": True,
                "geometry": LineString([(85.1, 27.9), [85.2, 27.9]]),
            }
        ],
        crs="EPSG:4326",
    )
    aff_path = tmp_path / "affected_roads.geojson"
    affected_roads_gdf.to_file(aff_path, driver="GeoJSON")

    run_dir = tmp_path / "run_stage5"

    result = run_stage5_pipeline(
        affected_roads=aff_path,
        osm_source=osm_data,
        settings=settings,
        run_dir=run_dir,
        event_date=event_date,
        data_mode=DataMode.SYNTHETIC_TEST,
    )

    # 1. Output files exist
    assert result.outputs["cut_off_settlements"].exists()
    assert result.outputs["settlement_connectivity"].exists()
    assert result.outputs["summary"].exists()
    assert result.outputs["provenance"].exists()

    # 2. Check summary schema and values
    summary = result.summary
    assert isinstance(summary, ConnectivitySummary)
    assert summary.total_settlements == 1  # 1 origin village (town is destination)
    assert summary.disconnected_count == 1
    assert summary.connected_count == 0
    assert summary.affected_road_segments_count == 1
    assert summary.osm_snapshot_date == snapshot_date
    assert summary.event_date == event_date

    # 3. Check GeoJSON contents
    cut_off_data = json.loads(result.outputs["cut_off_settlements"].read_text())
    assert len(cut_off_data["features"]) == 1
    feat = cut_off_data["features"][0]
    assert feat["properties"]["settlement_id"] == "node/2"
    assert feat["properties"]["status"] == "disconnected"
    assert "way/101" in feat["properties"]["disrupting_road_ids"]

    # 4. Check ProvenanceLog
    prov_data = json.loads(result.outputs["provenance"].read_text())
    assert prov_data["data_mode"] == DataMode.SYNTHETIC_TEST.value
    step_names = [s["name"] for s in prov_data["steps"]]
    assert "build_road_graph" in step_names
    assert "evaluate_connectivity" in step_names
    assert "write_connectivity_outputs" in step_names


def test_stage5_pipeline_data_rules_guard_post_event(tmp_path: Path, settings: Settings):
    event_date = date(2026, 8, 26)
    post_event_date = date(2026, 8, 28)  # Post-event OSM edit!

    osm_data = _make_mock_osm_data(post_event_date)
    affected_roads_gdf = gpd.GeoDataFrame(
        [{"osm_id": "way/1", "potentially_affected": True, "geometry": LineString([(85.1, 27.9), (85.2, 27.9)])}],
        crs="EPSG:4326",
    )

    with pytest.raises(DataRuleViolation, match="Post-event OSM edits are prohibited"):
        run_stage5_pipeline(
            affected_roads=affected_roads_gdf,
            osm_source=osm_data,
            settings=settings,
            run_dir=tmp_path / "post_event_run",
            event_date=event_date,
        )


def test_stage5_pipeline_prohibited_input_name(tmp_path: Path, settings: Settings):
    event_date = date(2026, 8, 26)
    snapshot_date = date(2026, 7, 27)
    osm_data = _make_mock_osm_data(snapshot_date)

    # Attempt to pass Copernicus EMS / EMSR as an input file
    prohibited_file = tmp_path / "emsr927_damage_assessment_roads.geojson"
    prohibited_gdf = gpd.GeoDataFrame(
        [{"osm_id": "way/1", "potentially_affected": True, "geometry": LineString([(85.1, 27.9), (85.2, 27.9)])}],
        crs="EPSG:4326",
    )
    prohibited_gdf.to_file(prohibited_file, driver="GeoJSON")

    with pytest.raises(DataRuleViolation, match="matches prohibited-source pattern"):
        run_stage5_pipeline(
            affected_roads=prohibited_file,
            osm_source=osm_data,
            settings=settings,
            run_dir=tmp_path / "prohibited_run",
            event_date=event_date,
        )
