#!/usr/bin/env python
"""Stage 3: Sentinel-1 -> Preprocessing -> Terrain -> (Change Detection + AI Segmentation) -> Fusion.

Example (real data):
    python scripts/run_stage3.py \\
        --aoi data/raw/dev/aoi_from_search.geojson \\
        --event-date 2026-08-26 \\
        --before data/raw/dev/s1_before_20260816.tif \\
        --after  data/raw/dev/s1_after_20260828.tif \\
        --dem    data/dem/dev/dem_glo30.tif \\
        --out    data/runs/dev_stage3

Outputs (in --out):
    preprocessed/ (s1_before_db.tif, s1_after_db.tif)
    terrain/ (slope_deg.tif, hand_m.tif, terrain_flags.tif)
    change/ (change_classes.tif, evidence_source.tif, ai_probability.tif,
             log_ratio_db.tif, exclusion_flags.tif, summary.json)
    quicklooks/*.png (change_classes.png, evidence_source.png, ai_probability.png, etc.)
    aoi.geojson
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
from floodmap.core.contracts import AnalysisRequest
from floodmap.core.errors import FloodmapError
from floodmap.core.geo_utils import load_aoi
from floodmap.pipeline.stage3_pipeline import run_stage3_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True,
                    help="GeoJSON file, s1_candidates.json or 'min_lon,min_lat,max_lon,max_lat'")
    ap.add_argument("--event-date", required=True, type=date.fromisoformat)
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--dem", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Run directory (default: <runs_dir>/<run-id>)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None, help="Optional override YAML")
    ap.add_argument("--no-quicklooks", action="store_true")
    args = ap.parse_args()

    missing = [f"--{name} {path}  (resolved: {Path(path).resolve()})"
               for name, path in (("before", args.before), ("after", args.after), ("dem", args.dem))
               if path is not None and not Path(path).is_file()]
    if missing:
        print("ERROR: input file(s) not found (working directory: " + str(Path.cwd()) + "):\n  "
              + "\n  ".join(missing), file=sys.stderr)
        return 2
    try:
        aoi = load_aoi(args.aoi)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    settings = load_settings(override_path=args.config)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    out = args.out or settings.resolve_path(settings.paths.runs_dir) / run_id
    req = AnalysisRequest(
        aoi=aoi,
        event_date=args.event_date,
        before_scene=args.before,
        after_scene=args.after,
        dem=args.dem,
    )
    try:
        res = run_stage3_pipeline(req, settings, out, analysis_id=run_id, write_quicklooks=not args.no_quicklooks)
    except FloodmapError as e:
        print(f"ERROR ({type(e).__name__}): {e}\nProvenance (if any): {out / 'provenance.json'}", file=sys.stderr)
        return 2

    s = res.summary
    print(f"\nStage 3 Run {s.analysis_id}  data_mode={s.data_mode.value}  method={s.method}")
    print(f"Grid {s.crs} @ {s.resolution_m:g} m | AOI {s.aoi_area_km2:.3f} km² | observable {s.observable_area_km2:.3f} km²")
    print(f"AI Model: {s.ai_model} (initial/uncalibrated threshold={s.ai_threshold:.2f})")
    print(f"AI Stats: water_pixels={s.ai_stats.get('water_pixels', 0)}, water_km²={s.ai_stats.get('water_km2', 0.0)}, mean_probability={s.ai_stats.get('mean_probability', 0.0)}")
    print(f"Fusion mode: {s.fusion_mode}")

    print("\nFinal Fused Class Areas (km²):")
    for k, v in s.class_areas_km2.items():
        print(f"  {k:32s} {v:10.4f}")

    print("\nEvidence Source Areas (km²):")
    for k, v in s.evidence_areas_km2.items():
        print(f"  {k:32s} {v:10.4f}")

    if s.class3_context_km2:
        print("\nNon-water change by terrain context (km²; labels only):")
        for k, v in s.class3_context_km2.items():
            print(f"  {k:32s} {v:10.4f}")

    print("\nExcluded / rejected inside AOI (km², flags may overlap):")
    for k, v in s.excluded_area_km2.items():
        print(f"  {k:32s} {v:10.4f}")

    if s.warnings:
        print("\nWarnings:")
        for w in s.warnings:
            print(f"  - {w}")

    print(f"\nOutputs written to: {res.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
