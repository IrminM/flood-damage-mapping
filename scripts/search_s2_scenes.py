#!/usr/bin/env python
"""Search the Copernicus catalogue for Sentinel-2 Level-1C scenes around an event.

Example:
    python scripts/search_s2_scenes.py --aoi config/aoi_examples/trishuli_corridor_dev.geojson \
        --event-date 2026-08-26 --out data/raw/dev/s2_candidates.json

No login needed (public OData catalogue). Outputs candidate scenes with cloud cover,
orbit, platform, and acquisition time.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from floodmap.core.geo_utils import load_aoi
from floodmap.data_sources.cdse_catalogue import search_s2_l1c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True, help="Path to AOI GeoJSON or polygon")
    ap.add_argument("--event-date", required=True, type=date.fromisoformat, help="Event date YYYY-MM-DD")
    ap.add_argument("--days-before", type=int, default=30, help="Days before event to search")
    ap.add_argument("--days-after", type=int, default=21, help="Days after event to search")
    ap.add_argument("--max-cloud", type=float, default=None, help="Filter scenes exceeding this cloud cover percentage")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = ap.parse_args()

    aoi = load_aoi(args.aoi)
    scenes = search_s2_l1c(aoi, args.event_date, days_before=args.days_before, days_after=args.days_after)
    if args.max_cloud is not None:
        scenes = [s for s in scenes if s.cloud_cover_pct is not None and s.cloud_cover_pct <= args.max_cloud]

    print(f"{len(scenes)} Sentinel-2 L1C scenes found:")
    for i, s in enumerate(scenes):
        cloud_str = f"cloud={s.cloud_cover_pct:.1f}%" if s.cloud_cover_pct is not None else "cloud=N/A"
        tile_part = s.product_ids[0].split("_")[-2] if len(s.product_ids[0].split("_")) > 2 else ""
        print(f"  [{i:2d}] {s.acquisition_time:%Y-%m-%d %H:%M} {s.platform:12s} "
              f"orbit={s.relative_orbit} {tile_part:6s} {cloud_str:12s} {s.product_ids[0]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aoi": aoi.model_dump(),
        "event_date": args.event_date.isoformat(),
        "scenes": [json.loads(s.model_dump_json()) for s in scenes],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote candidate metadata to {args.out}")


if __name__ == "__main__":
    main()
