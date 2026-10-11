#!/usr/bin/env python
"""Stage 6: DEM-Based Downstream Flood Path & Settlement Tracing.

Example:
    python scripts/run_stage6.py \\
        --dem data/dem/dev/dem_glo30.tif \\
        --lon 85.25 --lat 28.15 \\
        --osm-dir data/osm/dev \\
        --event-date 2026-08-26 \\
        --out data/runs/smoke_stage6/downstream

Outputs (in --out):
    downstream_path.geojson
    downstream_path_settlements.geojson
    downstream_path_summary.json
    provenance.json
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

from floodmap.core.config import load_settings
from floodmap.core.errors import FloodmapError
from floodmap.pipeline.stage6_pipeline import run_stage6_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True, type=Path, help="Path to Copernicus DEM GeoTIFF")
    ap.add_argument("--lon", type=float, default=None, help="Upstream starting longitude (decimal degrees)")
    ap.add_argument("--lat", type=float, default=None, help="Upstream starting latitude (decimal degrees)")
    ap.add_argument("--upstream-point", type=str, default=None, help="Upstream point as 'lon,lat' string")
    ap.add_argument("--upstream-file", type=Path, default=None, help="Path to GeoJSON Point file")
    ap.add_argument("--osm-dir", required=True, type=Path, help="Path to pre-event OSM directory")
    ap.add_argument("--event-date", required=True, type=date.fromisoformat, help="Disaster event date (YYYY-MM-DD)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--sink-handling", choices=["fill", "stop"], default=None, help="Sink/flat handling (default from config: fill)")
    ap.add_argument("--max-length-km", type=float, default=None, help="Maximum path length in km (default 50 km)")
    ap.add_argument("--proximity-radius-m", type=float, default=None, help="Settlement detection radius in meters (default 1000m)")
    ap.add_argument("--run-id", default=None, help="Optional run analysis ID")
    ap.add_argument("--config", default=None, help="Optional override YAML")
    args = ap.parse_args()

    if not args.dem.is_file():
        print(f"ERROR: DEM file not found: {args.dem}", file=sys.stderr)
        return 2
    if not args.osm_dir.is_dir():
        print(f"ERROR: OSM directory not found: {args.osm_dir}", file=sys.stderr)
        return 2

    # Parse upstream point from args
    if args.lon is not None and args.lat is not None:
        upstream_pt = (args.lon, args.lat)
    elif args.upstream_point:
        parts = args.upstream_point.replace(",", " ").split()
        if len(parts) < 2:
            print(f"ERROR: Invalid --upstream-point format: {args.upstream_point}. Expected 'lon,lat'", file=sys.stderr)
            return 2
        upstream_pt = (float(parts[0]), float(parts[1]))
    elif args.upstream_file:
        if not args.upstream_file.is_file():
            print(f"ERROR: Upstream file not found: {args.upstream_file}", file=sys.stderr)
            return 2
        upstream_pt = args.upstream_file
    else:
        print("ERROR: Upstream point must be provided via (--lon AND --lat), --upstream-point, or --upstream-file", file=sys.stderr)
        return 2

    settings = load_settings(override_path=args.config)
    if args.sink_handling:
        settings.downstream.sink_handling = args.sink_handling
    if args.max_length_km:
        settings.downstream.max_length_m = args.max_length_km * 1000.0
    if args.proximity_radius_m:
        settings.downstream.settlement_proximity_radius_m = args.proximity_radius_m

    run_id = args.run_id or uuid.uuid4().hex[:12]

    try:
        res = run_stage6_pipeline(
            dem_source=args.dem,
            upstream_point=upstream_pt,
            osm_source=args.osm_dir,
            settings=settings,
            run_dir=args.out,
            event_date=args.event_date,
            analysis_id=run_id,
        )
    except FloodmapError as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    s = res.summary
    print(f"\nStage 6 Downstream Flood Path Tracing: {s.analysis_id}")
    print(f"Upstream Input: lon={s.upstream_input_coords[0]}, lat={s.upstream_input_coords[1]}")
    print(f"Snapped To DEM: lon={s.upstream_snapped_coords[0]}, lat={s.upstream_snapped_coords[1]} (snapped {s.snapping_distance_m:.1f} m)")
    print(f"Elevation Profile: {s.start_elevation_m:.1f} m -> {s.end_elevation_m:.1f} m (drop: {s.total_elevation_drop_m:.1f} m)")
    print(f"Path Length: {s.path_length_km:.2f} km ({s.path_vertices_count} cells)")
    print(f"Termination: {s.termination_reason}")
    print(f"Nearby Settlements Encountered: {s.settlements_count}")

    if s.settlements:
        print("\n--- Chronological Settlement Sequence Along Path ---")
        for st in s.settlements:
            s_name = (st.settlement_name or st.settlement_id).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
            print(f"  {st.order}. {s_name} ({st.place_type}) "
                  f"- {st.distance_along_path_km:.2f} km downstream (offset {st.distance_to_path_m:.1f} m)")

    print("\nGenerated outputs:")
    for k, p in res.outputs.items():
        print(f"  {k}: {p}")

    if s.warnings:
        print(f"\nWarnings ({len(s.warnings)}):")
        for w in s.warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
