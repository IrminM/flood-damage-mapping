"""Baseline change detection against the SYNTHETIC_TEST valley scene (known truth)."""
import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from dev_mocks.synthetic_scene import make_synthetic_s1_pair, synthetic_aoi, synthetic_loaded_pair  # noqa: E402
from floodmap.core.contracts import ExclusionFlag, OutputClass  # noqa: E402
from floodmap.detection.interface import get_change_detector  # noqa: E402
from floodmap.satellite.grid import make_analysis_grid  # noqa: E402
from floodmap.satellite.preprocess import preprocess_s1_pair  # noqa: E402
from floodmap.satellite.reproject import reproject_to_grid  # noqa: E402
from floodmap.satellite.terrain import compute_terrain, look_geometry  # noqa: E402


def truth_on_grid(mask, src, grid):
    r = src.with_data(mask.astype(np.uint8), band_names=["t"], nodata=None)
    return reproject_to_grid(r, grid.spec, "nearest", dst_nodata=0).data[0] == 1


def iou(pred, truth):
    return (pred & truth).sum() / max((pred | truth).sum(), 1)


@pytest.fixture(scope="module")
def scene():
    from floodmap.core.config import load_settings
    settings = load_settings(environ={})
    pair = make_synthetic_s1_pair(seed=7)
    grid = make_analysis_grid(synthetic_aoi(pair, inset_m=30), 10, 0)
    b, a = synthetic_loaded_pair(pair)
    pp = preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)
    tcfg = settings.terrain.model_copy(update={"hand_stream_threshold_km2": 0.05, "margin_m": 0})
    tp = compute_terrain(pair.dem, grid, tcfg, look_geometry("ascending", None, tcfg))
    truths = {k: truth_on_grid(getattr(pair, k), pair.before, grid)
              for k in ("flood_truth", "permanent_water", "change_truth", "slope_false_water", "after_nodata")}
    return settings, pair, grid, pp, tp, truths


def test_detects_classes_against_known_truth(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    c = res.classes.data[0]
    assert res.classes.dtype if hasattr(res.classes, "dtype") else c.dtype == np.uint8
    assert set(np.unique(c)) <= {0, 1, 2, 3, 255}
    assert iou(c == OutputClass.FLOOD_WATER, t["flood_truth"]) > 0.8
    assert iou(c == OutputClass.PERMANENT_WATER, t["permanent_water"]) > 0.6
    assert iou(c == OutputClass.DEBRIS_OR_CHANGE, t["change_truth"]) > 0.7
    # false positives on untouched land are rare
    land = pp.valid & ~(t["flood_truth"] | t["permanent_water"] | t["change_truth"] | t["slope_false_water"])
    assert (c[land] != OutputClass.BACKGROUND).mean() < 0.01


def test_terrain_gate_rejects_dark_steep_slope(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    c, ex = res.classes.data[0], res.exclusion
    patch = t["slope_false_water"] & pp.valid
    assert patch.sum() > 100
    assert (c[patch] == OutputClass.FLOOD_WATER).mean() < 0.01
    assert ((ex[patch] & ExclusionFlag.TERRAIN_WATER_GATE) > 0).mean() > 0.8
    assert res.stats["rejected_by_terrain_px"]["water"] > 0


def test_without_dem_false_water_appears_and_is_warned(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, None)
    patch = t["slope_false_water"] & pp.valid
    assert (res.classes.data[0][patch] == OutputClass.FLOOD_WATER).mean() > 0.5
    assert any("No DEM" in w for w in res.warnings)


def test_unobservable_pixels_are_255_with_reasons(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    c, ex = res.classes.data[0], res.exclusion
    assert (c[~grid.aoi_mask] == 255).all() and ((ex[~grid.aoi_mask] & ExclusionFlag.OUTSIDE_AOI) > 0).all()
    nod = pp.nodata_in_aoi
    assert nod.any() and (c[nod] == 255).all() and ((ex[nod] & ExclusionFlag.NO_DATA) > 0).all()


def test_threshold_and_log_ratio_outputs(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    thr = res.thresholds["water_vv"]
    assert thr["method"] in ("otsu", "fixed_fallback") and -24 <= thr["value_db"] <= -14
    assert res.log_ratio_db.band_names == ["VV_log_ratio", "VH_log_ratio"]
    lr = res.log_ratio_db.band("VV_log_ratio")
    assert np.nanmedian(lr[t["flood_truth"] & pp.valid]) < -8          # ~ -12 dB drop
    assert np.nanmedian(lr[t["change_truth"] & pp.valid]) == pytest.approx(6.0, abs=1.5)   # x4 = +6 dB


def test_fixed_threshold_mode_and_min_area(scene):
    settings, pair, grid, pp, tp, t = scene
    s = settings.model_copy(deep=True)
    s.change_detection.s1_logratio_v1.water_threshold.mode = "fixed"
    s.change_detection.s1_logratio_v1.min_area_m2 = 1e9     # nothing survives
    res = get_change_detector(s).detect(pp, tp)
    assert res.thresholds["water_vv"]["method"] == "fixed"
    assert not np.isin(res.classes.data[0], [1, 2, 3]).any()
    assert ((res.exclusion & ExclusionFlag.SMALL_BLOB) > 0).any()


def test_unknown_method_rejected(settings):
    s = settings.model_copy(deep=True)
    s.change_detection.method = "does_not_exist"
    with pytest.raises(ValueError, match="available"):
        get_change_detector(s)


def test_change_diagnostics_do_not_alter_classes_and_carry_evidence(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    dg = res.stats["diagnostics"]
    # noise model: speckle-only land gives a finite robust sigma and a small expected exceedance
    for pol in ("VV", "VH"):
        n = dg["noise"][pol]
        assert 0 < n["robust_sigma_db"] < 3 and 0 <= n["gaussian_exceedance_lower_bound"] < 0.1
        assert n["observed_exceedance"] >= 0
    # the synthetic change patch brightens (+6 dB)
    assert dg["class3_sign"]["brightening_share"] > 0.95
    # the synthetic change patch lies on the flat floodplain (low HAND) -> enriched in the lowest band
    low = dg["class3_by_hand_m"][0]
    assert low["band"] == [0.0, 10.0] and low["enrichment"] > 1.5
    assert sum(r["share_of_class3"] for r in dg["class3_by_hand_m"]) == pytest.approx(1.0, abs=1e-3)
    # diagnostics are evidence only: class-3 area equals the class raster
    c3 = (res.classes.data[0] == OutputClass.DEBRIS_OR_CHANGE).sum()
    assert dg["final_class3_fraction"] == pytest.approx(c3 / (pp.valid & (res.classes.data[0] != 1) &
                                                             (res.classes.data[0] != 2)).sum(), rel=0.05)


def test_noise_only_pair_produces_little_class3(settings):
    """Null test: two speckle realisations of the SAME scene (no real change)."""
    p1 = make_synthetic_s1_pair(seed=11, nodata_cols=0)
    p2 = make_synthetic_s1_pair(seed=12, nodata_cols=0)
    grid = make_analysis_grid(synthetic_aoi(p1, inset_m=30), 10, 0)
    b, _ = synthetic_loaded_pair(p1)
    _, a = synthetic_loaded_pair(p1)
    a.raster = p2.before          # same land/river, independent speckle, no flood/change
    pp = preprocess_s1_pair(b, a, grid, settings.preprocessing.s1)
    res = get_change_detector(settings).detect(pp, None)
    dg = res.stats["diagnostics"]
    # speckle alone produces a few isolated exceedances (heavier tails than the Gaussian bound) ...
    assert dg["observed_candidates_fraction"] < 0.01
    assert dg["observed_candidates_fraction"] >= dg["gaussian_exceedance_any_pol_lower_bound"]
    # ... which the opening + minimum-area clean-up removes entirely: no change -> no class 3
    assert dg["final_class3_fraction"] < 0.001
    assert not (res.classes.data[0] == OutputClass.DEBRIS_OR_CHANGE).any()


# ------------------------------------------------------------------ noise-adaptive threshold & context

from floodmap.core.contracts import ChangeContext  # noqa: E402
from floodmap.detection.change_s1 import classify_change_context  # noqa: E402


def _noisy_scene(mode, looks=2):
    from floodmap.core.config import load_settings
    s = load_settings(environ={"FLOODMAP__CHANGE_DETECTION__S1_LOGRATIO_V1__CHANGE_THRESHOLD__MODE": mode})
    p = make_synthetic_s1_pair(seed=5, looks=looks, change_gain=40.0)        # +16 dB real change patch
    grid = make_analysis_grid(synthetic_aoi(p, inset_m=30), 10, 0)
    b, a = synthetic_loaded_pair(p)
    pp = preprocess_s1_pair(b, a, grid, s.preprocessing.s1)
    land = truth_on_grid(~(p.flood_truth | p.permanent_water | p.change_truth | p.slope_false_water | p.after_nodata),
                         p.before, grid) & pp.valid
    patch = truth_on_grid(p.change_truth, p.before, grid) & pp.valid
    return get_change_detector(s).detect(pp, None), land, patch


def test_noise_adaptive_threshold_cuts_false_change_and_keeps_real_change():
    fixed, land, patch = _noisy_scene("fixed")
    adapt, _, _ = _noisy_scene("noise_adaptive")
    fp_fixed = (fixed.classes.data[0][land] == 3).mean()
    fp_adapt = (adapt.classes.data[0][land] == 3).mean()
    thr = adapt.thresholds["change"]
    assert thr["mode"] == "noise_adaptive" and thr["VV"]["method"] == "noise_adaptive"
    assert thr["VV"]["value_db"] > 3.0 and thr["VV"]["value_db"] == pytest.approx(3 * thr["VV"]["robust_sigma_db"], rel=1e-3)
    assert fp_adapt < fp_fixed / 5                                     # far fewer false alarms on stable land
    assert (adapt.classes.data[0][patch] == 3).mean() > 0.95           # strong real change still detected
    assert adapt.stats["diagnostics"]["threshold_db"]["VV"] == pytest.approx(thr["VV"]["value_db"], abs=1e-3)


def test_clean_scene_uses_floor_so_results_are_unchanged(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    ch = res.thresholds["change"]
    assert ch["VV"]["method"] == "noise_adaptive_floor" and ch["VV"]["value_db"] == 3.0
    s_fixed = settings.model_copy(deep=True)
    s_fixed.change_detection.s1_logratio_v1.change_threshold.mode = "fixed"
    res_fixed = get_change_detector(s_fixed).detect(pp, tp)
    assert np.array_equal(res.classes.data[0], res_fixed.classes.data[0])


def test_classify_change_context_rules():
    change = np.array([[1, 1, 1, 1, 0]], bool)
    observable = np.array([[1, 1, 1, 1, 1]], bool)
    hand = np.array([[5.0, 30.0, 5.0, np.nan, 5.0]])
    slope = np.array([[3.0, 3.0, 25.0, 3.0, 3.0]])
    ctx = classify_change_context(change, observable, hand, slope, 25.0, 15.0)
    assert ctx.tolist() == [[ChangeContext.VALLEY_BOTTOM, ChangeContext.HILLSIDE, ChangeContext.HILLSIDE,
                             ChangeContext.UNKNOWN, ChangeContext.NOT_CHANGE]]
    no_dem = classify_change_context(change, observable, None, None, 25.0, 15.0)
    assert (no_dem[change] == ChangeContext.UNKNOWN).all()
    obs2 = observable.copy(); obs2[0, 4] = False
    assert classify_change_context(change, obs2, hand, slope, 25, 15)[0, 4] == ChangeContext.NODATA


def test_context_labels_do_not_change_classes_and_find_floodplain_patch(scene):
    settings, pair, grid, pp, tp, t = scene
    res = get_change_detector(settings).detect(pp, tp)
    c, ctx = res.classes.data[0], res.context
    is3 = c == OutputClass.DEBRIS_OR_CHANGE
    assert is3.any() and (np.isin(ctx[is3], [1, 2, 3])).all() and not np.isin(ctx[~is3], [1, 2, 3]).any()
    patch = t["change_truth"] & is3
    assert (ctx[patch] == ChangeContext.VALLEY_BOTTOM).mean() > 0.9     # synthetic patch sits on the floodplain
    off = settings.model_copy(deep=True)
    off.change_detection.s1_logratio_v1.change_context.enabled = False
    res_off = get_change_detector(off).detect(pp, tp)
    assert res_off.context is None and np.array_equal(res_off.classes.data[0], c)
