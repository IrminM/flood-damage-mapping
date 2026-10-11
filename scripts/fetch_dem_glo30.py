#!/usr/bin/env python
"""Download Copernicus DEM GLO-30 for an AOI (public AWS bucket, no login).

Example (PowerShell):
    python scripts\\fetch_dem_glo30.py --aoi config\\aoi_examples\\trishuli_corridor_dev.geojson --out data\\dem\\dev\\dem_glo30.tif

Writes the mosaicked/cropped DEM (EPSG:4326, float32, nodata -32767) + dem_glo30.meta.json
(source_type=copernicus-dem, tile list with SHA-256, bounds, AOI coverage). Tiles are cached
in <out folder>\\tiles and validated before reuse.

The default buffer covers the terrain-analysis margin (config terrain.margin_m), the grid
margin, plus 1 km, so drainage paths leaving the AOI are not cut short.
If an existing DEM already covers AOI + buffer it is kept (use --overwrite to rebuild).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from floodmap.core.config import load_settings
from floodmap.core.errors import FloodmapError
from floodmap.core.geo_utils import load_aoi
from floodmap.data_sources.copernicus_dem import existing_dem_covers, fetch_dem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True, help="GeoJSON file or 'min_lon,min_lat,max_lon,max_lat'")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--buffer-km", type=float, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true", help="Rebuild even if an existing DEM covers the AOI")
    ap.add_argument("--retries", type=int, default=4, help="Attempts per tile download (default 4)")
    args = ap.parse_args(argv)

    settings = load_settings(override_path=args.config)
    buffer_km = args.buffer_km if args.buffer_km is not None else \
        (settings.terrain.margin_m + settings.grid.margin_m) / 1000.0 + 1.0
    aoi = load_aoi(args.aoi)
    if not args.overwrite and existing_dem_covers(args.out, aoi, buffer_km):
        print(f"DEM already present and covers AOI + {buffer_km:g} km: {args.out} (use --overwrite to rebuild)")
        return 0
    try:
        out = fetch_dem(aoi, args.out, buffer_km, attempts=args.retries)
    except FloodmapError as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 2
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
