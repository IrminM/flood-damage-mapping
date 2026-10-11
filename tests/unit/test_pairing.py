from datetime import date

from floodmap.satellite.pairing import check_pair, rank_candidate_pairs

EVENT = date(2026, 8, 26)


def test_valid_same_track_pair(settings, s1_meta_factory):
    r = check_pair(s1_meta_factory("2026-08-20"), s1_meta_factory("2026-09-01"), EVENT, settings.satellite.s1)
    assert r.valid, r.errors
    assert r.gap_days == 12.0
    assert r.warnings == []


def test_different_relative_orbit_rejected(settings, s1_meta_factory):
    r = check_pair(s1_meta_factory("2026-08-20", orbit=85), s1_meta_factory("2026-09-01", orbit=12),
                   EVENT, settings.satellite.s1)
    assert not r.valid and any("relative orbit" in e for e in r.errors)


def test_direction_and_unknown_orbit_rejected(settings, s1_meta_factory):
    r = check_pair(s1_meta_factory("2026-08-20"), s1_meta_factory("2026-09-01", direction="descending"),
                   EVENT, settings.satellite.s1)
    assert not r.valid and any("direction" in e for e in r.errors)
    r = check_pair(s1_meta_factory("2026-08-20", relative_orbit=None), s1_meta_factory("2026-09-01"),
                   EVENT, settings.satellite.s1)
    assert not r.valid and any("unknown" in e.lower() for e in r.errors)


def test_dates_on_wrong_side_of_event(settings, s1_meta_factory):
    r = check_pair(s1_meta_factory("2026-08-08"), s1_meta_factory("2026-08-20"), EVENT, settings.satellite.s1)
    assert not r.valid and any("before the event" in e for e in r.errors)
    r = check_pair(s1_meta_factory("2026-08-26"), s1_meta_factory("2026-09-07"), EVENT, settings.satellite.s1)
    assert not r.valid


def test_event_day_after_image_warns(settings, s1_meta_factory):
    r = check_pair(s1_meta_factory("2026-08-14"), s1_meta_factory("2026-08-26"), EVENT, settings.satellite.s1)
    assert r.valid and any("event day" in w for w in r.warnings)


def test_gap_too_large_and_odd_gap(settings, s1_meta_factory):
    r = check_pair(s1_meta_factory("2026-07-15"), s1_meta_factory("2026-09-01"), EVENT, settings.satellite.s1)
    assert not r.valid and any("exceeds" in e for e in r.errors)
    r = check_pair(s1_meta_factory("2026-08-22"), s1_meta_factory("2026-09-01"), EVENT, settings.satellite.s1)
    assert r.valid and any("repeat cycle" in w for w in r.warnings)


def test_orbit_check_can_be_disabled_by_config(settings, s1_meta_factory):
    cfg = settings.satellite.s1.model_copy(update={"require_same_relative_orbit": False})
    r = check_pair(s1_meta_factory("2026-08-20", orbit=85), s1_meta_factory("2026-09-01", orbit=12), EVENT, cfg)
    assert r.valid


def test_rank_candidates_prefers_closest_valid_pair(settings, s1_meta_factory):
    scenes = [
        s1_meta_factory("2026-08-08"), s1_meta_factory("2026-08-20"),
        s1_meta_factory("2026-09-01"), s1_meta_factory("2026-09-13"),
        s1_meta_factory("2026-08-22", orbit=12), s1_meta_factory("2026-08-28", orbit=12, platform="sentinel-1c"),
    ]
    pairs = rank_candidate_pairs(scenes, EVENT, settings.satellite.s1)
    assert pairs, "expected at least one valid pair"
    b, a, res = pairs[0]
    # orbit 12 pair (08-22 -> 08-28) is closest to the event
    assert (str(b.acquisition_time.date()), str(a.acquisition_time.date())) == ("2026-08-22", "2026-08-28")
    assert all(p[0].relative_orbit == p[1].relative_orbit for p in pairs)
    assert all(p[2].valid for p in pairs)
