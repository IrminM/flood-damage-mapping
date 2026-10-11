import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, box
from pathlib import Path

from scripts.compare_emsr927_validation import (
    compute_raster_comparison_metrics,
    check_spatial_suitability,
    scan_for_reference_files,
    run_emsr927_validation,
)


def test_compute_raster_comparison_metrics_perfect_match():
    # 10x10 array
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


def test_compute_raster_comparison_metrics_partial_overlap():
    model = np.zeros((10, 10), dtype=np.uint8)
    model[0:4, 0:4] = 1  # 16 pixels

    ref = np.zeros((10, 10), dtype=np.uint8)
    ref[2:6, 2:6] = 1  # 16 pixels
    # Overlap (TP): rows 2:4, cols 2:4 -> 2x2 = 4 pixels
    # FP: 16 - 4 = 12 pixels
    # FN: 16 - 4 = 12 pixels
    roi = np.ones((10, 10), dtype=np.uint8)

    res = compute_raster_comparison_metrics(model, ref, roi, pixel_area_m2=100.0)
    assert res["pixels"]["tp"] == 4
    assert res["pixels"]["fp"] == 12
    assert res["pixels"]["fn"] == 12
    # Precision: 4 / (4 + 12) = 0.25
    # Recall: 4 / (4 + 12) = 0.25
    # Dice: 2*4 / (2*4 + 12 + 12) = 8 / 32 = 0.25
    # IoU: 4 / (4 + 12 + 12) = 4 / 28 = 0.1429
    assert res["metrics"]["precision"] == 0.25
    assert res["metrics"]["recall"] == 0.25
    assert res["metrics"]["dice_f1"] == 0.25
    assert round(res["metrics"]["iou_jaccard"], 4) == 0.1429


def test_spatial_suitability_check():
    study_bounds = (85.17, 27.95, 85.36, 28.18)

    # Polygon inside study bounds (Syapru Besi: ~85.33, 28.15)
    inside_poly = Polygon([(85.25, 28.05), (85.30, 28.05), (85.30, 28.10), (85.25, 28.10)])
    gdf_inside = gpd.GeoDataFrame(geometry=[inside_poly], crs="EPSG:4326")
    res_in = check_spatial_suitability(gdf_inside, study_bounds)
    assert res_in["intersects"] is True
    assert res_in["overlap_pct"] > 99.0

    # Polygon completely outside (e.g., Kathmandu valley / Pokhara: 84.0, 28.2)
    outside_poly = Polygon([(83.0, 28.0), (83.5, 28.0), (83.5, 28.5), (83.0, 28.5)])
    gdf_outside = gpd.GeoDataFrame(geometry=[outside_poly], crs="EPSG:4326")
    res_out = check_spatial_suitability(gdf_outside, study_bounds)
    assert res_out["intersects"] is False
    assert res_out["overlap_pct"] == 0.0


def test_run_emsr927_validation_missing_data(tmp_path):
    empty_dir = tmp_path / "emsr927_empty"
    empty_dir.mkdir()
    res = run_emsr927_validation(ref_dir=empty_dir)
    assert res["status"] == "AWAITING_REFERENCE_DATA"
    assert "activation" in res
