"""End-to-end stage-3 pipeline integration tests on synthetic scenes."""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from dev_mocks.synthetic_scene import synthetic_aoi, write_synthetic_pair
from floodmap.core.config import load_settings
from floodmap.core.contracts import (
    AnalysisRequest,
    DataMode,
    EvidenceSource,
    OutputClass,
    Stage3Summary,
)
from floodmap.pipeline.stage3_pipeline import run_stage3_pipeline

EVENT = date(2026, 8, 26)
SMALL_SCENE_ENV = {
    "FLOODMAP__TERRAIN__HAND_STREAM_THRESHOLD_KM2": "0.05",
    "FLOODMAP__TERRAIN__MARGIN_M": "0",
    "FLOODMAP__GRID__MARGIN_M": "0",
    "FLOODMAP__AI__S1_MINI_UNET_V1__TILE_SIZE": "128",
    "FLOODMAP__AI__S1_MINI_UNET_V1__OVERLAP": "16",
}


@pytest.fixture
def synth_pair(tmp_path):
    bpath, apath, pair = write_synthetic_pair(tmp_path / "in")
    aoi = synthetic_aoi(pair, inset_m=30)
    return bpath, apath, tmp_path / "in" / "SYNTHETIC_dem.tif", pair, aoi


def test_stage3_pipeline_end_to_end(synth_pair, tmp_path):
    bpath, apath, dpath, pair, aoi = synth_pair
    req = AnalysisRequest(
        aoi=aoi,
        event_date=EVENT,
        before_scene=bpath,
        after_scene=apath,
        dem=dpath,
    )
    settings = load_settings(environ=SMALL_SCENE_ENV)
    run_dir = tmp_path / "stage3_run"

    res = run_stage3_pipeline(req, settings, run_dir, analysis_id="s3_test")
    rd = res.run_dir

    expected_files = [
        "preprocessed/s1_before_db.tif",
        "preprocessed/s1_after_db.tif",
        "change/change_classes.tif",
        "change/evidence_source.tif",
        "change/ai_probability.tif",
        "change/log_ratio_db.tif",
        "change/exclusion_flags.tif",
        "change/summary.json",
        "terrain/slope_deg.tif",
        "terrain/hand_m.tif",
        "terrain/terrain_flags.tif",
        "quicklooks/change_classes.png",
        "quicklooks/evidence_source.png",
        "quicklooks/ai_probability.png",
        "quicklooks/before_vv_db.png",
        "aoi.geojson",
        "provenance.json",
    ]
    for rel in expected_files:
        assert (rd / rel).exists(), f"Missing expected output: {rel}"

    # Validate Summary contract
    summary = Stage3Summary.model_validate_json((rd / "change/summary.json").read_text(encoding="utf-8"))
    assert summary.data_mode is DataMode.SYNTHETIC_TEST
    assert summary.ai_model == "s1_mini_unet_v1"
    assert summary.ai_threshold == 0.50
    assert summary.fusion_mode == "union"

    # Validate rasters match summary
    with rasterio.open(rd / "change/change_classes.tif") as src_cls:
        cls_data = src_cls.read(1)
        assert src_cls.nodata == int(OutputClass.NODATA)
        px_km2 = abs(src_cls.transform.a * src_cls.transform.e) / 1e6

    with rasterio.open(rd / "change/evidence_source.tif") as src_ev:
        ev_data = src_ev.read(1)
        assert src_ev.nodata == int(EvidenceSource.NONE)

    with rasterio.open(rd / "change/ai_probability.tif") as src_prob:
        prob_data = src_prob.read(1)
        assert np.isnan(src_prob.nodata) or src_prob.nodata is None or src_prob.nodata != 0.0

    aoi_mask = res.grid.aoi_mask

    # Check evidence area reconciliation
    flood_water_km2 = ((cls_data == int(OutputClass.FLOOD_WATER)) & aoi_mask).sum() * px_km2
    assert summary.class_areas_km2["new_open_water"] == pytest.approx(flood_water_km2, rel=1e-5)
    assert summary.evidence_areas_km2["total_flood"] == pytest.approx(flood_water_km2, rel=1e-5)

    ev_both_km2 = ((ev_data == int(EvidenceSource.BOTH)) & aoi_mask).sum() * px_km2
    assert summary.evidence_areas_km2["confirmed_by_both"] == pytest.approx(ev_both_km2, rel=1e-5)

    # Validate Provenance Log contains Stage 3 steps
    prov_data = json.loads((rd / "provenance.json").read_text(encoding="utf-8"))
    step_names = [s["name"] for s in prov_data.get("steps", [])]
    assert "preprocess_s1" in step_names
    assert "change_detection" in step_names
    assert "ai_segmentation" in step_names
    assert "evidence_fusion" in step_names


def test_stage3_pipeline_without_dem(synth_pair, tmp_path):
    bpath, apath, _, _, aoi = synth_pair
    req = AnalysisRequest(
        aoi=aoi,
        event_date=EVENT,
        before_scene=bpath,
        after_scene=apath,
        dem=None,
    )
    settings = load_settings(environ=SMALL_SCENE_ENV)
    run_dir = tmp_path / "stage3_no_dem"

    res = run_stage3_pipeline(req, settings, run_dir, analysis_id="s3_no_dem")
    assert res.terrain is None
    assert any("No DEM" in w for w in res.warnings)
    assert (res.run_dir / "change/change_classes.tif").exists()
    assert (res.run_dir / "change/evidence_source.tif").exists()
