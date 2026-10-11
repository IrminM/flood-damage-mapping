"""Execute Stage-5 Road Network Connectivity and Cut-Off Analysis on Trishuli Scene.

Inputs:
  - Pre-event OSM: data/osm/dev/ (snapshot 2026-07-27)
  - Stage 4 Affected Roads: data/runs/infrastructure_impact_full_scene/affected_roads.geojson

Outputs written to data/runs/connectivity_analysis_full_scene/:
  1. connectivity_results.json
  2. disconnected_settlements.geojson
  3. road_network.geojson
  4. connectivity_summary.json
  5. connectivity_provenance.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

# Ensure floodmap package is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from floodmap.core.config import load_settings
from floodmap.core.contracts import DataMode
from floodmap.pipeline.stage5_pipeline import run_stage5_pipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OSM_DIR = Path("data/osm/dev")
DEFAULT_AFFECTED_ROADS = Path("data/runs/infrastructure_impact_full_scene/affected_roads.geojson")
DEFAULT_OUT_DIR = Path("data/runs/connectivity_analysis_full_scene")
EVENT_DATE = date(2026, 8, 26)


def run_stage5(
    osm_dir: Path = DEFAULT_OSM_DIR,
    affected_roads_path: Path = DEFAULT_AFFECTED_ROADS,
    out_dir: Path = DEFAULT_OUT_DIR,
    event_date: date = EVENT_DATE,
    disruption_mode: str = "remove",
) -> None:
    print("=" * 80)
    print("STAGE 5: ROAD NETWORK CONNECTIVITY & CUT-OFF ANALYSIS (TRISHULI SCENE)")
    print("=" * 80)
    print(f"OSM Pre-Event Source:    {osm_dir}")
    print(f"Stage 4 Affected Roads:  {affected_roads_path}")
    print(f"Event Date:              {event_date.isoformat()}")
    print(f"Disruption Mode:         {disruption_mode}")
    print(f"Output Directory:        {out_dir}")

    t0 = time.time()
    settings = load_settings()

    print("\n[1/3] Executing Stage 5 pipeline...")
    result = run_stage5_pipeline(
        affected_roads=affected_roads_path,
        osm_source=osm_dir,
        settings=settings,
        run_dir=out_dir,
        event_date=event_date,
        analysis_id="connectivity_trishuli",
        aoi_name="trishuli_event_aoi",
        data_mode=DataMode.REAL,
        disruption_mode=disruption_mode,
    )

    print("\n[2/3] Stage 5 Deliverables Written:")
    for name in [
        "connectivity_results",
        "disconnected_settlements",
        "road_network",
        "connectivity_summary",
        "connectivity_provenance",
    ]:
        p = result.outputs.get(name)
        if p and p.exists():
            print(f"  Written {name:28s} -> {p}")

    s = result.summary
    print("\n[3/3] Connectivity Analysis Summary Metrics:")
    print("-" * 80)
    print(f"Total Settlements Evaluated:     {s.total_settlements}")
    print(f"  1. CONNECTED:                  {s.connected_count} ({100.0 * s.connected_count / max(s.total_settlements, 1):.1f}%)")
    print(f"     - With Alternate Detour:    {s.alternate_route_count}")
    print(f"  2. DISCONNECTED:               {s.disconnected_count} ({100.0 * s.disconnected_count / max(s.total_settlements, 1):.1f}%)")
    print(f"  3. NO_BASELINE_ROUTE:          {s.no_baseline_route_count} (isolated road component prior to event)")
    print(f"  4. OUTSIDE_NETWORK:            {s.outside_network_count} (> {settings.connectivity.max_snapping_distance_m:.0f}m from mapped road)")
    print(f"Affected Road Segments Handled:  {s.affected_road_segments_count}")
    print(f"Disrupted Graph Components:      {s.disconnected_components_count}")
    print(f"Isolated Components (No Svc):    {s.isolated_components_count}")

    # Display 5 examples of disconnected settlements
    disc_recs = [r for r in json.loads(result.outputs["connectivity_results"].read_text(encoding="utf-8"))["settlements"] if r.get("classification") == "DISCONNECTED"]
    if disc_recs:
        print("\nExample Disconnected Settlements:")
        for r in disc_recs[:5]:
            name = r.get("settlement_name") or "Unnamed"
            sid = r.get("settlement_id")
            ptype = r.get("place_type")
            dest = r.get("nearest_destination_name")
            b_dist = f"{r.get('baseline_distance_m', 0) / 1000.0:.2f} km" if r.get("baseline_distance_m") else "N/A"
            roads = r.get("disrupting_road_ids") or []
            tiers = r.get("disrupting_evidence_tiers") or []
            print(f"  * {name} ({ptype}, {sid}): baseline to '{dest}' ({b_dist}) cut by {len(roads)} road(s): {roads} [{', '.join(tiers)}]")

    elapsed = time.time() - t0
    print(f"\nStage 5 execution completed in {elapsed:.2f} seconds.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 road network connectivity analysis.")
    parser.add_argument("--osm-dir", type=Path, default=DEFAULT_OSM_DIR, help="Path to pre-event OSM directory")
    parser.add_argument("--affected-roads", type=Path, default=DEFAULT_AFFECTED_ROADS, help="Path to Stage 4 affected roads GeoJSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--disruption-mode", type=str, default="remove", choices=["remove", "penalty"], help="Network disruption mode")
    args = parser.parse_args()

    run_stage5(
        osm_dir=args.osm_dir,
        affected_roads_path=args.affected_roads,
        out_dir=args.out_dir,
        disruption_mode=args.disruption_mode,
    )


if __name__ == "__main__":
    main()
