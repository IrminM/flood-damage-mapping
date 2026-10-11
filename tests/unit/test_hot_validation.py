import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, box
from pathlib import Path

from scripts.compare_hot_validation import (
    compute_raster_comparison_metrics,
    check_spatial_suitability,
    run_hot_validation,
    DEFAULT_HOT_FILE,
)


def test_hot_compute_raster_comparison_metrics_perfect_match():
    model = np.zeros((10, 10), dtype=np.uint8)
    model[2:5, 2:5] = 1  # 9 pixels of water
    ref = np.zeros((10, 10), dtype=np.uint8)
    ref[2:5, 2:5] = 1  # exact same 9 pixels
    roi = np.ones((10, 10), dtype=np.uint8)

    res = compute_raster_comparison_metrics(model, ref, roi, pixel_area_m2=100.0)
    assert res["pixels"]["tp"] == 9
    assert res["pixels"]["fp"] == 0
    assert res["pixels"]["fn"] == 0
    assert res["metrics"]["precision"] == 1.0
    assert res["metrics"]["recall"] == 1.0
    assert res["metrics"]["dice_f1"] == 1.0
    assert res["metrics"]["iou_jaccard"] == 1.0


def test_hot_compute_raster_comparison_metrics_partial_overlap():
    model = np.zeros((10, 10), dtype=np.uint8)
    model[0:4, 0:4] = 1  # 16 pixels

    ref = np.zeros((10, 10), dtype=np.uint8)
    ref[2:6, 2:6] = 1  # 16 pixels
    roi = np.ones((10, 10), dtype=np.uint8)

    res = compute_raster_comparison_metrics(model, ref, roi, pixel_area_m2=100.0)
    assert res["pixels"]["tp"] == 4
    assert res["pixels"]["fp"] == 12
    assert res["pixels"]["fn"] == 12
    assert res["metrics"]["precision"] == 0.25
    assert res["metrics"]["recall"] == 0.25
    assert res["metrics"]["dice_f1"] == 0.25
    assert round(res["metrics"]["iou_jaccard"], 4) == 0.1429


def test_hot_spatial_suitability_check():
    study_bounds = (85.17, 27.95, 85.36, 28.18)

    inside_poly = Polygon([(85.25, 28.05), (85.30, 28.05), (85.30, 28.10), (85.25, 28.10)])
    gdf_inside = gpd.GeoDataFrame(geometry=[inside_poly], crs="EPSG:4326")
    res_in = check_spatial_suitability(gdf_inside, study_bounds)
    assert res_in["intersects"] is True
    assert res_in["overlap_pct_of_total_ref"] > 99.0
    assert res_in["ref_is_valid"] is True

    outside_poly = Polygon([(83.0, 28.0), (83.5, 28.0), (83.5, 28.5), (83.0, 28.5)])
    gdf_outside = gpd.GeoDataFrame(geometry=[outside_poly], crs="EPSG:4326")
    res_out = check_spatial_suitability(gdf_outside, study_bounds)
    assert res_out["intersects"] is False
    assert res_out["overlap_pct_of_total_ref"] == 0.0


def test_hot_validation_missing_file(tmp_path):
    missing_file = tmp_path / "non_existent.geojson"
    res = run_hot_validation(ref_file=missing_file)
    assert res["status"] == "AWAITING_REFERENCE_DATA"
    assert "reference_info" in res


def test_hot_validation_end_to_end(tmp_path):
    # Tests the actual quarantined file if present
    if not DEFAULT_HOT_FILE.exists():
        pytest.skip("Quarantined HOT reference file not present.")

    out_dir = tmp_path / "test_val_hot"
    res = run_hot_validation(
        ref_file=DEFAULT_HOT_FILE,
        runs_dir=Path("data/runs"),
        output_dir=out_dir,
    )

    assert "layers_evaluated" in res
    assert "terrain_gated_ai_water" in res["layers_evaluated"]
    assert "fused_dual_evidence" in res["layers_evaluated"]
    assert res["reference"]["source"] == "Humanitarian OpenStreetMap Team (HOT)"

    # Verify metrics consistency
    gated = res["layers_evaluated"]["terrain_gated_ai_water"]
    assert gated["reference_water_area_km2"] == pytest.approx(8.906, rel=1e-2)
    assert gated["metrics"]["precision"] == pytest.approx(0.4898, abs=0.01)

    # Verify output files written
    assert (out_dir / "validation_metrics.json").exists()
    assert (out_dir / "validation_report.md").exists()
