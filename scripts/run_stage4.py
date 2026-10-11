#!/usr/bin/env python
"""Stage 4: Pre-event OSM Data + Raster Flood/Change Evidence -> Infrastructure Impact Assessment.

Example:
    python scripts/run_stage4.py \\
        --classes data/runs/smoke_stage3_real/change/change_classes.tif \\
        --evidence data/runs/smoke_stage3_real/change/evidence_source.tif \\
        --osm-dir data/osm/dev \\
        --event-date 2026-08-26 \\
        --out data/runs/smoke_stage3_real/impact

Outputs (in --out):
    infrastructure_summary.json
    affected_buildings.geojson
    affected_roads.geojson
    affected_bridges.geojson
    flood_evidence_vector.geojson
    provenance.json
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

import rasterio

import _bootstrap  # noqa: F401

from floodmap.core.config import load_settings
from floodmap.core.contracts import DataMode
from floodmap.core.errors import FloodmapError
from floodmap.core.georaster import GeoRaster
from floodmap.pipeline.stage4_pipeline import run_stage4_pipeline


def _load_raster(path: Path) -> GeoRaster:
    with rasterio.open(path) as src:
        data = src.read()
        transform = tuple(src.transform)[:6]
        crs = str(src.crs)
        nodata = src.nodata
    return GeoRaster(data=data, transform=transform, crs=crs, nodata=nodata)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", required=True, type=Path, help="Path to change_classes.tif")
    ap.add_argument("--evidence", type=Path, default=None, help="Path to evidence_source.tif (optional)")
    ap.add_argument("--osm-dir", required=True, type=Path, help="Path to pre-event OSM directory")
    ap.add_argument("--event-date", required=True, type=date.fromisoformat)
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None, help="Optional override YAML")
    args = ap.parse_args()

    if not args.classes.is_file():
        print(f"ERROR: classes raster not found: {args.classes}", file=sys.stderr)
        return 2
    if not args.osm_dir.is_dir():
        print(f"ERROR: OSM directory not found: {args.osm_dir}", file=sys.stderr)
        return 2

    settings = load_settings(override_path=args.config)
    classes_r = _load_raster(args.classes)
    evidence_r = _load_raster(args.evidence) if args.evidence and args.evidence.is_file() else None

    run_id = args.run_id or uuid.uuid4().hex[:12]
    try:
        res = run_stage4_pipeline(
            classes_raster=classes_r,
            osm_source=args.osm_dir,
            settings=settings,
            run_dir=args.out,
            event_date=args.event_date,
            evidence_raster=evidence_r,
            analysis_id=run_id,
        )
    except FloodmapError as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    s = res.summary
    print(f"\nStage 4 Infrastructure Impact Assessment: {s.analysis_id}")
    print(f"OSM snapshot date: {s.osm_snapshot_date} | Event date: {s.event_date}")
    print("\n--- Buildings ---")
    print(f"Total buildings in AOI: {s.buildings.total_count}")
    print(f"Potentially affected: {s.buildings.potentially_affected_count} ({s.buildings.affected_pct:.1f}%)")
    print(f"Affected footprint: {s.buildings.affected_footprint_m2:.1f} m² of {s.buildings.total_footprint_m2:.1f} m²")
    if s.buildings.by_class:
        print(f"By class: {s.buildings.by_class}")
    if s.buildings.by_evidence:
        print(f"By evidence: {s.buildings.by_evidence}")

    print("\n--- Roads ---")
    print(f"Total road network: {s.roads.total_length_km:.2f} km ({s.roads.total_segments} segments)")
    print(f"Potentially affected: {s.roads.potentially_affected_length_km:.2f} km ({s.roads.affected_pct:.1f}%, {s.roads.potentially_affected_segments} segments)")
    if s.roads.by_highway_type_km:
        print("By highway type:")
        for hwy, d in s.roads.by_highway_type_km.items():
            print(f"  {hwy:15s}: {d['affected_km']:.2f} km affected / {d['total_km']:.2f} km total")
    if s.roads.by_class_km:
        print(f"By class: {s.roads.by_class_km}")

    print("\n--- Bridges ---")
    print(f"Total bridges in AOI: {s.bridges.total_count}")
    print(f"Potentially affected: {s.bridges.potentially_affected_count} ({s.bridges.affected_pct:.1f}%)")
    if s.bridges.by_class:
        print(f"By class: {s.bridges.by_class}")

    if s.waterways_context:
        print("\n--- Waterway Context ---")
        print(f"Pre-event waterways: {s.waterways_context.total_waterways_km:.2f} km")
        print(f"Class 3 non-water change within {s.waterways_context.buffer_distance_m:.0f}m: {s.waterways_context.class3_near_waterway_km2:.4f} km²")
        print(f"Class 3 away from waterways: {s.waterways_context.class3_away_from_waterway_km2:.4f} km²")

    print(f"\nOutputs written to: {res.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
