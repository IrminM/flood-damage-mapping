"""End-to-end stage-2 pipeline on SYNTHETIC_TEST files (real files go through the same code)."""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from dev_mocks.synthetic_scene import synthetic_aoi, write_synthetic_pair  # noqa: E402
from floodmap.core.config import load_settings  # noqa: E402
from floodmap.core.contracts import AnalysisRequest, ChangeSummary, DataMode  # noqa: E402
from floodmap.core.errors import DataRuleViolation, ScenePairError  # noqa: E402
from floodmap.data_sources.copernicus_dem import tile_name, tile_url, tiles_for_bounds  # noqa: E402
from floodmap.pipeline.change_pipeline import run_change_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVENT = date(2026, 8, 26)
# Synthetic extent is only ~2 x 1.5 km, so drainage threshold and margins are scaled down.
SMALL_SCENE_ENV = {
    "FLOODMAP__TERRAIN__HAND_STREAM_THRESHOLD_KM2": "0.05",
    "FLOODMAP__TERRAIN__MARGIN_M": "0",
    "FLOODMAP__GRID__MARGIN_M": "0",
}


@pytest.fixture
def synth(tmp_path):
    bpath, apath, pair = write_synthetic_pair(tmp_path / "in")
    aoi = synthetic_aoi(pair, inset_m=30)
    return bpath, apath, tmp_path / "in" / "SYNTHETIC_dem.tif", pair, aoi


def run(synth, tmp_path, dem=True, **kw):
    bpath, apath, dpath, pair, aoi = synth
    req = AnalysisRequest(aoi=aoi, event_date=EVENT, before_scene=bpath, after_scene=apath,
                          dem=dpath if dem else None)
    return run_change_pipeline(req, load_settings(environ=SMALL_SCENE_ENV), tmp_path / "run", analysis_id="t1", **kw)


def test_end_to_end_outputs_and_summary(synth, tmp_path):
    res = run(synth, tmp_path)
    rd = res.run_dir
    for rel in ["preprocessed/s1_before_db.tif", "preprocessed/s1_after_db.tif", "change/change_classes.tif",
                "change/log_ratio_db.tif", "change/exclusion_flags.tif", "change/summary.json",
                "terrain/slope_deg.tif", "terrain/hand_m.tif", "terrain/terrain_flags.tif",
                "quicklooks/change_classes.png", "quicklooks/before_vv_db.png", "aoi.geojson", "provenance.json"]:
        assert (rd / rel).exists(), rel

    summary = ChangeSummary.model_validate_json((rd / "change/summary.json").read_text())
    assert summary.data_mode is DataMode.SYNTHETIC_TEST            # synthetic inputs are never reported as REAL
    assert summary.terrain_applied and summary.crs == "EPSG:32645"

    # every reported area is recomputable from the written raster
    with rasterio.open(rd / "change/change_classes.tif") as src:
        cls = src.read(1)
        assert src.nodata == 255 and cls.dtype == np.uint8
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
    aoi_mask = res.grid.aoi_mask
    for code, key in [(1, "new_open_water"), (2, "water_both_dates"), (3, "non_water_backscatter_change")]:
        assert summary.class_areas_km2[key] == pytest.approx(((cls == code) & aoi_mask).sum() * px_km2, rel=1e-6)

    # new-water area close to the synthetic truth (1.5 km x ... patch = 0.117 km²)
    _, _, _, pair, _ = synth
    truth_km2 = pair.flood_truth.sum() * 100 / 1e6
    assert summary.class_areas_km2["new_open_water"] == pytest.approx(truth_km2, rel=0.25)
    assert summary.excluded_area_km2["rejected_water_by_terrain_gate"] > 0
    # terrain context of class 3: written, labels only, and areas add up to the class-3 area
    assert (rd / "change/change_context.tif").exists() and (rd / "quicklooks/change_context.png").exists()
    ctx = summary.class3_context_km2
    assert set(ctx) == {"valley_bottom", "hillside", "unknown"}
    assert sum(ctx.values()) == pytest.approx(summary.class_areas_km2["non_water_backscatter_change"], abs=1e-6)
    assert ctx["valley_bottom"] > ctx["hillside"]
    assert summary.thresholds["change"]["mode"] == "noise_adaptive"
    assert any("NOT a debris map" in n for n in summary.interpretation_notes)


def test_provenance_records_every_step(synth, tmp_path):
    res = run(synth, tmp_path)
    prov = json.loads(res.provenance_path.read_text())
    assert prov["data_mode"] == "SYNTHETIC_TEST"
    steps = [s["name"] for s in prov["steps"]]
    assert steps == ["load_inputs", "pair_check", "analysis_grid", "preprocess_s1", "terrain",
                     "change_detection", "write_outputs"]
    assert all(s["status"] == "ok" for s in prov["steps"])
    roles = {i["role"]: i for i in prov["inputs"]}
    assert set(roles) == {"before", "after", "dem"} and all(len(i["sha256"]) == 64 for i in prov["inputs"])
    cd = next(s for s in prov["steps"] if s["name"] == "change_detection")
    assert cd["params"]["method"] == "s1_logratio_v1" and "min_vv_drop_db" in cd["params"]
    assert "water_vv" in cd["outputs"][0]["thresholds"]


def test_without_dem_runs_and_warns(synth, tmp_path):
    res = run(synth, tmp_path, dem=False, write_quicklooks=False)
    assert not res.summary.terrain_applied
    assert any("No DEM" in w for w in res.summary.warnings)
    assert not (res.run_dir / "terrain").exists()
    assert not (res.run_dir / "quicklooks").exists()


def test_invalid_pair_stops_and_logs(synth, tmp_path):
    bpath, apath, *_ = synth
    side = apath.with_suffix(".meta.json")
    meta = json.loads(side.read_text())
    meta["relative_orbit"] = 99
    side.write_text(json.dumps(meta))
    with pytest.raises(ScenePairError, match="relative orbits"):
        run(synth, tmp_path)
    prov = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert prov["steps"][-1]["name"] == "pair_check" and prov["steps"][-1]["status"] == "failed"


def test_prohibited_input_rejected(tmp_path):
    bpath, apath, pair = write_synthetic_pair(tmp_path / "EMSR927_maps")
    req = AnalysisRequest(aoi=synthetic_aoi(pair, 30), event_date=EVENT, before_scene=bpath, after_scene=apath)
    with pytest.raises(DataRuleViolation):
        run_change_pipeline(req, load_settings(environ=SMALL_SCENE_ENV), tmp_path / "run")


def test_cli_runs_on_files(synth, tmp_path):
    bpath, apath, dpath, pair, aoi = synth
    aoi_file = tmp_path / "aoi.geojson"
    aoi_file.write_text(json.dumps({"type": "Feature", "properties": {"name": "SYNTHETIC_AOI"},
                                    "geometry": aoi.geometry}))
    override = tmp_path / "small.yaml"
    override.write_text("terrain:\n  hand_stream_threshold_km2: 0.05\n  margin_m: 0\ngrid:\n  margin_m: 0\n")
    out = tmp_path / "cli_run"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/run_change_detection.py"), "--aoi", str(aoi_file),
                        "--event-date", "2026-08-26", "--before", str(bpath), "--after", str(apath),
                        "--dem", str(dpath), "--out", str(out), "--config", str(override), "--no-quicklooks"],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert "data_mode=SYNTHETIC_TEST" in p.stdout and "new_open_water" in p.stdout
    assert (out / "change/summary.json").exists()


def test_dem_tile_names():
    assert tile_name(27, 85) == "Copernicus_DSM_COG_10_N27_00_E085_00_DEM"
    assert tile_name(-1, -70) == "Copernicus_DSM_COG_10_S01_00_W070_00_DEM"
    assert tile_url("X").endswith("/X/X.tif")
    assert tiles_for_bounds((85.1, 27.9, 85.4, 28.2)) == [tile_name(27, 85), tile_name(28, 85)]
    assert tiles_for_bounds((84.9, 27.2, 85.0, 27.3)) == [tile_name(27, 84)]   # max exactly on a tile edge


def test_dem_sidecar_source_type(synth, guard):
    from floodmap.core.errors import InputDataError
    from floodmap.data_sources.dem_local import LocalDEMProvider
    _, _, dpath, _, _ = synth
    assert LocalDEMProvider(guard).load_dem(dpath).metadata["source_type"] == "synthetic-test"
    dpath.with_suffix(".meta.json").write_text('{"source_type": "sentinel-1"}')
    with pytest.raises(InputDataError):
        LocalDEMProvider(guard).load_dem(dpath)


def test_cli_reports_missing_inputs_clearly(synth, tmp_path):
    bpath, apath, dpath, pair, aoi = synth
    p = subprocess.run([sys.executable, str(ROOT / "scripts/run_change_detection.py"),
                        "--aoi", "data\\raw\\dev\\aoi_from_search.geojson", "--event-date", "2026-08-26",
                        "--before", str(bpath), "--after", str(apath), "--out", str(tmp_path / "x")],
                       capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert p.returncode == 2 and "AOI file not found" in p.stderr and "could not convert" not in p.stderr
    p = subprocess.run([sys.executable, str(ROOT / "scripts/run_change_detection.py"),
                        "--aoi", "85.1,27.9,85.3,28.1", "--event-date", "2026-08-26",
                        "--before", "nope_before.tif", "--after", str(apath), "--out", str(tmp_path / "y")],
                       capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert p.returncode == 2 and "--before nope_before.tif" in p.stderr and "working directory" in p.stderr
    assert not (tmp_path / "y").exists()                   # nothing is written when inputs are missing


def test_cli_accepts_search_candidates_as_aoi(synth, tmp_path):
    bpath, apath, dpath, pair, aoi = synth
    cand = tmp_path / "s1_candidates.json"
    cand.write_text(json.dumps({"aoi": aoi.model_dump(), "event_date": "2026-08-26", "scenes": [], "pairs": []}))
    override = tmp_path / "small.yaml"
    override.write_text("terrain:\n  hand_stream_threshold_km2: 0.05\n  margin_m: 0\ngrid:\n  margin_m: 0\n")
    p = subprocess.run([sys.executable, str(ROOT / "scripts/run_change_detection.py"), "--aoi", str(cand),
                        "--event-date", "2026-08-26", "--before", str(bpath), "--after", str(apath),
                        "--dem", str(dpath), "--out", str(tmp_path / "run"), "--config", str(override),
                        "--no-quicklooks"], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    s = json.loads((tmp_path / "run/change/summary.json").read_text())
    assert s["aoi_name"] == aoi.name
