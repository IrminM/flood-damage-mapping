#!/usr/bin/env python
"""Stage 5: Cut-Off Settlement / Road Network Connectivity Analysis.

Example:
    python scripts/run_stage5.py \\
        --affected-roads data/runs/smoke_stage3_real/impact/affected_roads.geojson \\
        --osm-dir data/osm/dev \\
        --event-date 2026-08-26 \\
        --out data/runs/smoke_stage3_real/connectivity

Outputs (in --out):
    cut_off_settlements.geojson
    settlement_connectivity.geojson
    connectivity_summary.json
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
from floodmap.pipeline.stage5_pipeline import run_stage5_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affected-roads", required=True, type=Path, help="Path to Stage 4 affected_roads.geojson")
    ap.add_argument("--osm-dir", required=True, type=Path, help="Path to pre-event OSM directory")
    ap.add_argument("--event-date", required=True, type=date.fromisoformat, help="Disaster event date (YYYY-MM-DD)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--disruption-mode", choices=["remove", "penalty"], default=None, help="Disruption mode: remove | penalty")
    ap.add_argument("--run-id", default=None, help="Optional run analysis ID")
    ap.add_argument("--config", default=None, help="Optional override YAML")
    args = ap.parse_args()

    if not args.affected_roads.is_file():
        print(f"ERROR: affected roads file not found: {args.affected_roads}", file=sys.stderr)
        return 2
    if not args.osm_dir.is_dir():
        print(f"ERROR: OSM directory not found: {args.osm_dir}", file=sys.stderr)
        return 2

    settings = load_settings(override_path=args.config)
    run_id = args.run_id or uuid.uuid4().hex[:12]

    try:
        res = run_stage5_pipeline(
            affected_roads=args.affected_roads,
            osm_source=args.osm_dir,
            settings=settings,
            run_dir=args.out,
            event_date=args.event_date,
            analysis_id=run_id,
            disruption_mode=args.disruption_mode,
        )
    except FloodmapError as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 2

    s = res.summary
    print(f"\nStage 5 Road Network Connectivity Analysis: {s.analysis_id}")
    print(f"OSM snapshot date: {s.osm_snapshot_date} | Event date: {s.event_date}")
    print(f"Disruption mode: {s.disruption_mode}")
    print(f"Target destination types: {s.destination_types}")
    print(f"Affected road segments evaluated: {s.affected_road_segments_count}")
    print("\n--- Settlement Connectivity ---")
    print(f"Total settlements evaluated: {s.total_settlements}")
    print(f"Connected settlements: {s.connected_count}")
    print(f"Disconnected / cut-off settlements: {s.disconnected_count}")
    print(f"Settlements with insufficient network evidence: {s.insufficient_evidence_count}")

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
