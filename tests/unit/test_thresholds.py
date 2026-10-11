import numpy as np
import pytest

from floodmap.detection.thresholds import otsu, water_threshold


def bimodal(n_land=20000, n_water=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-11, 1.5, n_land), rng.normal(-23, 1.5, n_water)])


def test_otsu_splits_bimodal_distribution():
    t = otsu(bimodal())
    assert -20 < t < -14


def test_otsu_needs_variation():
    with pytest.raises(ValueError):
        otsu(np.full(10, 3.0))


def test_auto_otsu_within_bounds():
    r = water_threshold(bimodal(), "auto_otsu", -18, (-24, -14), 1000)
    assert r.method == "otsu" and -20 < r.value_db < -14 and r.raw_otsu_db == r.value_db
    assert set(r.as_dict()) >= {"value_db", "method", "raw_otsu_db", "ashman_d", "dark_fraction", "note"}


def bright_dark_land(seed=1):
    # land only, with bright (facing radar) and dark (facing away) slopes: Otsu splits LAND,
    # landing near -8 dB, far outside the open-water range (the real Trishuli run gave -7.61 dB)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-11, 1, 5000), rng.normal(-5, 1, 5000)])


def test_out_of_range_otsu_falls_back_to_conservative_fixed_by_default():
    r = water_threshold(bright_dark_land(), "auto_otsu", -18, (-24, -14), 1000)
    assert r.method == "fixed_fallback" and r.value_db == -18          # NOT the permissive -14 bound
    assert r.raw_otsu_db > -14 and "not a land/open-water split" in r.note


def test_clamp_only_as_explicit_policy():
    r = water_threshold(bright_dark_land(), "auto_otsu", -18, (-24, -14), 1000, out_of_bounds="clamp")
    assert r.method == "otsu_clamped" and r.value_db == -14
    with pytest.raises(ValueError):
        water_threshold(bright_dark_land(), "auto_otsu", -18, (-24, -14), 1000, out_of_bounds="whatever")


def test_unimodal_histogram_rejected_by_bimodality_check():
    rng = np.random.default_rng(2)
    v = rng.normal(-18, 2, 20000)          # unimodal, centred in the water range: Otsu ~ -18 but meaningless
    r = water_threshold(v, "auto_otsu", -19, (-24, -14), 1000)
    assert -24 <= r.raw_otsu_db <= -14 and r.ashman_d < 3.5
    assert r.method == "fixed_fallback" and r.value_db == -19 and "not clearly bimodal" in r.note


def test_small_water_fraction_falls_back():
    # narrow river: ~2 % water -> global Otsu cannot isolate it -> conservative fixed threshold
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(-11, 2, 19600), rng.normal(-22, 1.5, 400)])
    assert water_threshold(v, "auto_otsu", -18, (-24, -14), 1000).method == "fixed_fallback"


def test_clear_land_water_mix_is_accepted_with_evidence():
    r = water_threshold(bimodal(), "auto_otsu", -18, (-24, -14), 1000)
    assert r.method == "otsu" and r.ashman_d > 5 and 0.1 < r.dark_fraction < 0.25


def test_fallback_and_fixed():
    r = water_threshold(np.array([-20.0, -10.0, np.nan]), "auto_otsu", -18, (-24, -14), 1000)
    assert r.method == "fixed_fallback" and r.value_db == -18 and r.n_samples == 2
    assert water_threshold(bimodal(), "fixed", -17.5, (-24, -14), 1000).value_db == -17.5
    with pytest.raises(ValueError):
        water_threshold(bimodal(), "magic", -18, (-24, -14), 1000)


# ------------------------------------------------------------------ change threshold

from floodmap.detection.thresholds import change_threshold, robust_sigma  # noqa: E402


def test_change_threshold_fixed_and_adaptive():
    rng = np.random.default_rng(0)
    noisy = rng.normal(0, 2.0, 50000)            # sigma 2 dB -> 3 sigma = 6 dB
    clean = rng.normal(0, 0.5, 50000)            # sigma 0.5 dB -> 1.5 dB < floor
    assert change_threshold(noisy, "fixed", 3.0, 3.0, 1000).value_db == 3.0
    r = change_threshold(noisy, "noise_adaptive", 3.0, 3.0, 1000)
    assert r.method == "noise_adaptive" and r.value_db == pytest.approx(6.0, rel=0.03)
    r = change_threshold(clean, "noise_adaptive", 3.0, 3.0, 1000)
    assert r.method == "noise_adaptive_floor" and r.value_db == 3.0 and r.robust_sigma_db == pytest.approx(0.5, rel=0.03)
    r = change_threshold(noisy[:100], "noise_adaptive", 3.0, 3.0, 1000)
    assert r.method == "fixed_fallback" and r.value_db == 3.0
    with pytest.raises(ValueError):
        change_threshold(noisy, "magic", 3.0, 3.0, 1000)
    with pytest.raises(ValueError):
        change_threshold(noisy, "noise_adaptive", 3.0, 0.0, 1000)


def test_robust_sigma_tolerates_real_change():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0, 1.5, 90000), rng.normal(8, 1, 10000)])   # 10 % strong real change
    s = robust_sigma(v)
    # MAD is biased upward by one-sided real change (~+14 % here): the threshold errs conservative,
    # but stays far below the plain standard deviation, which real change inflates to ~3 dB
    assert 1.5 <= s < 1.5 * 1.2
    assert np.std(v) > 2.5 and s < 0.65 * np.std(v)
