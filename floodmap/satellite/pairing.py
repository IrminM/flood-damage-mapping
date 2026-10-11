"""Before/after scene pairing rules for change detection.

From the Track B brief: "compare images from the same orbit track (same time of day,
12 days apart). Images from different tracks view the terrain from different angles
and cannot be compared pixel by pixel."

`check_pair` validates a given pair; `rank_candidate_pairs` picks valid pairs from a
list of available scenes (e.g. from scripts/search_s1_scenes.py), closest to the event first.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Sequence

from ..core.config import S1Config
from ..core.contracts import PairCheckResult, SceneMetadata


def _days(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def _event_start(event_date: date, tz) -> datetime:
    return datetime.combine(event_date, time(0, 0), tzinfo=tz)


def check_pair(before: SceneMetadata, after: SceneMetadata, event_date: date, cfg: S1Config) -> PairCheckResult:
    errors: list[str] = []
    warnings: list[str] = []
    tb, ta = before.acquisition_time, after.acquisition_time
    if (tb.tzinfo is None) != (ta.tzinfo is None):
        tb, ta = tb.replace(tzinfo=None), ta.replace(tzinfo=None)
    tz = tb.tzinfo or (timezone.utc if ta.tzinfo else None)
    ev = _event_start(event_date, tz)

    if before.sensor != after.sensor:
        errors.append(f"Different sensors: before={before.sensor}, after={after.sensor}")

    if tb.date() >= event_date:
        errors.append(f"'Before' image ({tb.date()}) is not before the event date ({event_date})")
    if ta.date() < event_date:
        errors.append(f"'After' image ({ta.date()}) is before the event date ({event_date})")
    elif ta.date() == event_date:
        warnings.append("'After' image was acquired on the event day; it may predate the event")

    gap = _days(tb, ta)
    if gap <= 0:
        errors.append("'After' image is not later than 'before' image")

    if before.sensor == "S1" and after.sensor == "S1":
        if cfg.require_same_relative_orbit:
            if before.relative_orbit is None or after.relative_orbit is None:
                errors.append("Relative orbit unknown for one or both scenes; cannot confirm same track")
            elif before.relative_orbit != after.relative_orbit:
                errors.append(f"Different relative orbits ({before.relative_orbit} vs {after.relative_orbit}): "
                              "viewing geometry differs, pixel-wise comparison is invalid")
        if cfg.require_same_orbit_direction:
            if before.orbit_direction is None or after.orbit_direction is None:
                errors.append("Orbit direction unknown for one or both scenes")
            elif before.orbit_direction != after.orbit_direction:
                errors.append(f"Different orbit directions ({before.orbit_direction} vs {after.orbit_direction})")
        if gap > cfg.max_pair_gap_days:
            errors.append(f"Gap of {gap:.1f} days exceeds max_pair_gap_days={cfg.max_pair_gap_days}")
        elif gap > 0 and abs(gap - round(gap / 6) * 6) > 1.0:
            warnings.append(f"Gap of {gap:.1f} days is not a multiple of the 6/12-day repeat cycle; "
                            "check that both scenes are on the same track")
        missing = [b for b in cfg.expected_bands if b not in before.bands or b not in after.bands]
        if missing:
            warnings.append(f"Expected bands missing in one or both scenes: {missing}")
        if before.platform != after.platform:
            warnings.append(f"Different platforms ({before.platform} vs {after.platform}); "
                            "acceptable on the same track, but note possible calibration differences")
        if before.units != after.units:
            errors.append(f"Different units ({before.units} vs {after.units})")

    before_d = _days(tb, ev)
    after_d = _days(ev, ta)
    if before_d > cfg.warn_days_from_event:
        warnings.append(f"'Before' image is {before_d:.1f} days before the event (seasonal change possible)")
    if after_d > cfg.warn_days_from_event:
        warnings.append(f"'After' image is {after_d:.1f} days after the event (water may have receded)")

    return PairCheckResult(valid=not errors, errors=errors, warnings=warnings, gap_days=round(gap, 3),
                           before_days_from_event=round(before_d, 3), after_days_from_event=round(after_d, 3))


def rank_candidate_pairs(
    scenes: Sequence[SceneMetadata], event_date: date, cfg: S1Config
) -> list[tuple[SceneMetadata, SceneMetadata, PairCheckResult]]:
    """All valid (before, after) pairs, best first: closest to the event, then fewest warnings."""
    befores = [s for s in scenes if s.acquisition_time.date() < event_date]
    afters = [s for s in scenes if s.acquisition_time.date() >= event_date]
    pairs = []
    for b in befores:
        for a in afters:
            res = check_pair(b, a, event_date, cfg)
            if res.valid:
                pairs.append((b, a, res))
    pairs.sort(key=lambda p: ((p[2].before_days_from_event or 0) + (p[2].after_days_from_event or 0),
                              len(p[2].warnings)))
    return pairs
