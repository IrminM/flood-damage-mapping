#!/usr/bin/env python
"""Download a PRE-EVENT OpenStreetMap snapshot for an AOI from the ohsome API.

Example:
    python scripts/fetch_osm_ohsome.py \
        --aoi config/aoi_examples/my_aoi.geojson \
        --event-date 2026-08-26 \
        --snapshot-date 2026-07-27 \
        --out data/osm/my_aoi

    # or a bounding box instead of a file:
    python scripts/fetch_osm_ohsome.py --aoi 85.15,27.95,85.35,28.15 --event-date 2026-08-26 --out data/osm/test

Writes buildings/roads/bridges/places/health .geojson + snapshot_meta.json.
The snapshot date must be before the event date (enforced) — post-event OSM edits
are prohibited inputs. Filters and buffers come from config/default.yaml (osm.*).
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from floodmap.core.config import load_settings
from floodmap.core.data_rules import DataRulesGuard
from floodmap.core.geo_utils import load_aoi
from floodmap.data_sources.ohsome import fetch_snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True, help="GeoJSON file or 'min_lon,min_lat,max_lon,max_lat'")
    ap.add_argument("--event-date", required=True, type=date.fromisoformat)
    ap.add_argument("--snapshot-date", type=date.fromisoformat, default=None,
                    help="OSM snapshot date (default: config osm.default_snapshot_date)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--api-key", default=None, help="HeiGIT ohsome API key (or set OHSOME_API_KEY env var)")
    ap.add_argument("--config", default=None, help="Optional override YAML")
    args = ap.parse_args()

    settings = load_settings(override_path=args.config)
    snapshot = args.snapshot_date or date.fromisoformat(settings.osm.default_snapshot_date)
    aoi = load_aoi(args.aoi)
    out = fetch_snapshot(
        aoi,
        args.event_date,
        snapshot,
        args.out,
        settings.osm,
        DataRulesGuard(settings.data_rules),
        api_key=args.api_key,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
