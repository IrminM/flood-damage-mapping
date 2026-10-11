#!/usr/bin/env python
"""Stage 2: Sentinel-1 before/after -> preprocessing -> terrain masks -> change raster.

Example (real data):
    python scripts/run_change_detection.py \
        --aoi config/aoi_examples/trishuli_corridor_dev.geojson \
        --event-date 2026-08-26 \
        --before data/raw/dev/s1_before_20260820.tif \
        --after  data/raw/dev/s1_after_20260901.tif \
        --dem    data/dem/dev/dem_glo30.tif \
        --out    data/runs/dev_stage2

Each scene needs its .meta.json sidecar (written by fetch_s1_openeo.py). The DEM is
optional but strongly recommended (without it no shadow/layover/slope/HAND masking).

Outputs (in --out): preprocessed/, terrain/, change/ (change_classes.tif, log_ratio_db.tif,
exclusion_flags.tif, summary.json), quicklooks/*.png, aoi.geojson, provenance.json
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
from floodmap.pipeline.change_pipeline import run_change_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True,
                    help="GeoJSON file, s1_candidates.json (reuses the search AOI) or 'min_lon,min_lat,max_lon,max_lat'")
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
    req = AnalysisRequest(aoi=aoi, event_date=args.event_date,
                          before_scene=args.before, after_scene=args.after, dem=args.dem)
    try:
        res = run_change_pipeline(req, settings, out, analysis_id=run_id, write_quicklooks=not args.no_quicklooks)
    except FloodmapError as e:
        print(f"ERROR ({type(e).__name__}): {e}\nProvenance (if any): {out / 'provenance.json'}", file=sys.stderr)
        return 2

    s = res.summary
    print(f"\nRun {s.analysis_id}  data_mode={s.data_mode.value}  method={s.method}")
    print(f"Grid {s.crs} @ {s.resolution_m:g} m | AOI {s.aoi_area_km2:.3f} km² | observable {s.observable_area_km2:.3f} km²")
    wt = s.thresholds["water_vv"]
    raw = "n/a" if wt.get("raw_otsu_db") is None else f"{wt['raw_otsu_db']:.2f} dB"
    dval = "n/a" if wt.get("ashman_d") is None else f"{wt['ashman_d']:.2f}"
    print(f"Water threshold: {wt['value_db']:.2f} dB ({wt['method']}; raw Otsu {raw}, Ashman D {dval})")
    ch = s.thresholds.get("change", {})
    if ch:
        parts = [f"{pol} {ch[pol]['value_db']:.2f} dB ({ch[pol]['method']}"
                 + (f", sigma {ch[pol]['robust_sigma_db']:.2f}" if ch[pol].get("robust_sigma_db") is not None else "")
                 + ")" for pol in ("VV", "VH") if pol in ch]
        print(f"Change threshold [{ch['mode']}, k={ch['k_sigma']:g}, floor {ch['floor_db']:g} dB]: " + "; ".join(parts))
    print("Class areas (km²):")
    for k, v in s.class_areas_km2.items():
        print(f"  {k:32s} {v:10.4f}")
    if s.class3_context_km2:
        print("Non-water change by terrain context (km²; labels only):")
        for k, v in s.class3_context_km2.items():
            print(f"  {k:32s} {v:10.4f}")
    print("Excluded / rejected inside AOI (km², flags may overlap):")
    for k, v in s.excluded_area_km2.items():
        print(f"  {k:32s} {v:10.4f}")
    dg = s.detection_stats.get("diagnostics", {})
    if dg:
        thr_txt = ", ".join(f"{k} +/-{v:g} dB" for k, v in dg["threshold_db"].items())
        print(f"Non-water change diagnostics (thresholds {thr_txt}; classes unchanged by this):")
        for pol, n in dg.get("noise", {}).items():
            print(f"  {pol}: robust sigma {n['robust_sigma_db']:.2f} dB | |log-ratio| >= threshold observed "
                  f"{n['observed_exceedance']:.2%} (Gaussian noise lower bound {n['gaussian_exceedance_lower_bound']:.2%})")
        print(f"  candidates before clean-up {dg['observed_candidates_fraction']:.2%} of non-water area; "
              f"final class 3 {dg['final_class3_fraction']:.2%}")
        sg = dg["class3_sign"]
        print(f"  class 3 brightening {sg['brightening_share']:.0%} / darkening {sg['darkening_share']:.0%}")
        for row in dg.get("class3_by_hand_m", []):
            band = row["band"] if isinstance(row["band"], str) else \
                f"{row['band'][0]:g}-{row['band'][1]:g} m" if row["band"][1] is not None else f">{row['band'][0]:g} m"
            enr = "n/a" if row["enrichment"] is None else f"{row['enrichment']:.2f}"
            print(f"  HAND {band:>10s}: {row['class3_km2']:8.3f} km² class 3 | "
                  f"{row['share_of_class3']:.0%} of class 3 vs {row['share_of_area']:.0%} of area -> enrichment {enr}")
    if s.warnings:
        print("Warnings:")
        for w in s.warnings:
            print(f"  - {w}")
    print(f"\nOutputs in {res.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
