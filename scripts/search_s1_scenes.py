#!/usr/bin/env python
"""Search the Copernicus catalogue for Sentinel-1 GRD scenes around an event and
suggest valid SAME-TRACK before/after pairs.

Example:
    python scripts/search_s1_scenes.py --aoi config/aoi_examples/my_aoi.geojson \
        --event-date 2026-08-26 --out data/raw/my_aoi/s1_candidates.json

No login needed (public catalogue). The JSON output is the input for
scripts/fetch_s1_openeo.py, so scene metadata (relative orbit, direction, time)
flows into the sidecar files without manual typing.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from floodmap.core.config import load_settings
from floodmap.core.geo_utils import load_aoi
from floodmap.data_sources.cdse_catalogue import search_s1_grd
from floodmap.satellite.pairing import rank_candidate_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--event-date", required=True, type=date.fromisoformat)
    ap.add_argument("--days-before", type=int, default=30)
    ap.add_argument("--days-after", type=int, default=21)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    settings = load_settings(override_path=args.config)
    aoi = load_aoi(args.aoi)
    scenes = search_s1_grd(aoi, args.event_date, days_before=args.days_before, days_after=args.days_after)
    print(f"{len(scenes)} Sentinel-1 GRD scenes found")
    for i, s in enumerate(scenes):
        print(f"  [{i:2d}] {s.acquisition_time:%Y-%m-%d %H:%M} {s.platform:12s} "
              f"orbit={s.relative_orbit} {s.orbit_direction}  {s.product_ids[0]}")

    pairs = rank_candidate_pairs(scenes, args.event_date, settings.satellite.s1)
    print(f"\n{len(pairs)} valid same-track pairs (best first):")
    for i, (b, a, res) in enumerate(pairs[:10]):
        print(f"  pair {i}: {b.acquisition_time:%Y-%m-%d} -> {a.acquisition_time:%Y-%m-%d} "
              f"orbit {b.relative_orbit} {b.orbit_direction}, gap {res.gap_days:.1f} d"
              + (f"  warnings: {res.warnings}" if res.warnings else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aoi": aoi.model_dump(),
        "event_date": args.event_date.isoformat(),
        "scenes": [json.loads(s.model_dump_json()) for s in scenes],
        "pairs": [{"before": json.loads(b.model_dump_json()), "after": json.loads(a.model_dump_json()),
                   "check": res.model_dump()} for b, a, res in pairs],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
