"""Unit tests for the EMSR927 post-hoc infrastructure comparison runner."""
import json
from pathlib import Path
import pytest
import geopandas as gpd
from shapely.geometry import box, LineString, Point, Polygon

from scripts.compare_emsr927_infrastructure import (
    compute_shared_evaluation_area,
    compare_road_infrastructure,
    compare_bridge_infrastructure,
    compare_building_context,
    run_infrastructure_comparison,
    DEFAULT_REF_DIR,
    DEFAULT_RUNS_DIR,
)


def test_compute_shared_evaluation_area():
    # Two overlapping kilometer-scale boxes
    poly1 = box(0, 0, 1000, 1000)  # Area 1,000,000 m^2 = 1.0 km^2
    poly2 = box(500, 0, 1500, 1000)  # Area 1,000,000 m^2, intersection area 500,000 m^2 = 0.5 km^2

    gdf1 = gpd.GeoDataFrame(geometry=[poly1], crs="EPSG:32645")
    gdf2 = gpd.GeoDataFrame(geometry=[poly2], crs="EPSG:32645")

    shared_gdf, meta = compute_shared_evaluation_area(gdf1, gdf2, target_crs="EPSG:32645")
    assert len(shared_gdf) == 1
    assert meta["shared_evaluation_area_km2"] == pytest.approx(0.5, rel=1e-3)
    assert meta["overlap_percentage_of_emsr927"] == pytest.approx(50.0, rel=1e-3)


def test_compute_shared_evaluation_area_no_intersection():
    poly1 = box(0, 0, 10, 10)
    poly2 = box(20, 20, 30, 30)

    gdf1 = gpd.GeoDataFrame(geometry=[poly1], crs="EPSG:32645")
    gdf2 = gpd.GeoDataFrame(geometry=[poly2], crs="EPSG:32645")

    with pytest.raises(ValueError, match="do not intersect"):
        compute_shared_evaluation_area(gdf1, gdf2, target_crs="EPSG:32645")


def test_compare_road_infrastructure():
    shared_poly = box(0, 0, 1000, 1000)
    shared_gdf = gpd.GeoDataFrame(geometry=[shared_poly], crs="EPSG:32645")

    # Reference roads: 2 lines
    ref_line1 = LineString([(100, 100), (100, 500)])  # length 400m, Destroyed
    ref_line2 = LineString([(200, 100), (200, 500)])  # length 400m, No visible damage
    ref_gdf = gpd.GeoDataFrame(
        {"damage_gra": ["Destroyed", "No visible damage"]},
        geometry=[ref_line1, ref_line2],
        crs="EPSG:32645",
    )

    # Pipeline roads: 1 line overlapping ref_line1 within 10m
    pipe_line1 = LineString([(105, 100), (105, 500)])  # length 400m, affected
    pipe_line2 = LineString([(500, 100), (500, 500)])  # length 400m, unaffected
    pipe_gdf = gpd.GeoDataFrame(
        {"is_affected": [True, False]},
        geometry=[pipe_line1, pipe_line2],
        crs="EPSG:32645",
    )

    res = compare_road_infrastructure(
        ref_roads_gdf=ref_gdf,
        pipeline_roads_gdf=pipe_gdf,
        shared_eval_gdf=shared_gdf,
        buffer_distances_m=(15.0, 30.0),
        target_crs="EPSG:32645",
    )

    assert res["emsr927_reference_roads"]["total_segments_count"] == 2
    assert res["emsr927_reference_roads"]["damage_breakdown"]["Destroyed"]["length_m"] == 400.0
    assert res["floodintel_pipeline_roads"]["flagged_exposed_segments"] == 1
    assert res["floodintel_pipeline_roads"]["flagged_exposed_length_m"] == 400.0

    # Spatial coincidence: ref_line1 is 5m away from pipe_line1, so fully inside 15m buffer
    coinc15 = res["spatial_coincidence"]["destroyed_within_15m_buffer_of_floodintel_exposed"]
    assert coinc15["overlap_length_m"] == 400.0
    assert coinc15["pct_of_destroyed_roads"] == 100.0


def test_compare_bridge_infrastructure():
    shared_poly = box(0, 0, 1000, 1000)
    shared_gdf = gpd.GeoDataFrame(geometry=[shared_poly], crs="EPSG:32645")

    # Reference bridge at (100, 100)
    ref_pt = Point(100, 100)
    ref_gdf = gpd.GeoDataFrame(
        {"damage_gra": ["Destroyed"]},
        geometry=[ref_pt],
        crs="EPSG:32645",
    )

    # Pipeline bridge at (102, 100) -> distance 2m, flagged is_affected=True
    pipe_pt = Point(102, 100)
    pipe_gdf = gpd.GeoDataFrame(
        {"bridge_id": ["BR-01"], "is_affected": [True]},
        geometry=[pipe_pt],
        crs="EPSG:32645",
    )

    res = compare_bridge_infrastructure(
        ref_bridges_gdf=ref_gdf,
        pipeline_bridges_gdf=pipe_gdf,
        shared_eval_gdf=shared_gdf,
        target_crs="EPSG:32645",
    )

    assert res["emsr927_reference_bridges"]["total_in_shared_aoi"] == 1
    assert res["emsr927_reference_bridges"]["destroyed_count"] == 1
    assert res["floodintel_pipeline_bridges"]["flagged_exposed_count"] == 1
    assert res["spatial_coincidence"]["coincidence_percentage"] == 100.0
    assert len(res["spatial_coincidence"]["detailed_bridge_matches"]) == 1
    match0 = res["spatial_coincidence"]["detailed_bridge_matches"][0]
    assert match0["distance_to_nearest_pipeline_bridge_m"] == 2.0
    assert match0["floodintel_hazard_exposed"] is True


def test_compare_building_context():
    shared_poly = box(0, 0, 1000, 1000)
    shared_gdf = gpd.GeoDataFrame(geometry=[shared_poly], crs="EPSG:32645")

    ref_gdf = gpd.GeoDataFrame(
        {"damage_gra": ["Destroyed", "Damaged", "Possibly damaged"]},
        geometry=[Point(50, 50), Point(60, 60), Point(70, 70)],
        crs="EPSG:32645",
    )

    pipe_gdf = gpd.GeoDataFrame(
        {"is_affected": [True, False]},
        geometry=[box(45, 45, 55, 55), box(150, 150, 160, 160)],
        crs="EPSG:32645",
    )

    res = compare_building_context(ref_gdf, pipe_gdf, shared_gdf, target_crs="EPSG:32645")
    assert res["emsr927_point_structures"]["total_in_shared_aoi"] == 3
    assert res["floodintel_osm_building_polygons"]["total_in_shared_aoi"] == 2
    assert res["floodintel_osm_building_polygons"]["flagged_exposed_count"] == 1


def test_end_to_end_emsr927_infrastructure_comparison(tmp_path):
    # Verify execution against real repository outputs if present
    if not (DEFAULT_REF_DIR / "EMSR927_AOI01_GRA_PRODUCT_areaOfInterestA_v1.json").exists():
        pytest.skip("Quarantined EMSR927 extracted files not found.")

    out_dir = tmp_path / "val_infra_out"
    res = run_infrastructure_comparison(
        ref_dir=DEFAULT_REF_DIR,
        runs_dir=DEFAULT_RUNS_DIR,
        output_dir=out_dir,
        run_id="smoke_stage3_real",
    )

    # Core guardrails
    assert res["validation_status_preserved"] == "not_validated_on_Trishuli"
    assert "dice_f1" not in res
    assert "iou_jaccard" not in res
    assert "precision" not in res

    # Area metrics
    assert res["shared_evaluation_area"]["shared_evaluation_area_km2"] == pytest.approx(2.95, rel=1e-2)

    # Road metrics
    assert res["road_comparison"]["emsr927_reference_roads"]["damage_breakdown"]["Destroyed"]["length_km"] == pytest.approx(4.66, rel=1e-2)
    assert res["road_comparison"]["floodintel_pipeline_roads"]["flagged_exposed_length_km"] == pytest.approx(4.44, rel=1e-2)

    # Bridge metrics
    assert res["bridge_comparison"]["emsr927_reference_bridges"]["destroyed_pct"] == 100.0
    assert res["bridge_comparison"]["spatial_coincidence"]["coincidence_percentage"] == 100.0

    # Artifact generation
    assert (out_dir / "infrastructure_comparison_metrics.json").exists()
    assert (out_dir / "infrastructure_comparison_report.md").exists()
