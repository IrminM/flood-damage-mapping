"""Execute Stage-4 Infrastructure Impact Assessment on Trishuli Scene.

Intersects pre-event OpenStreetMap infrastructure (buildings, roads, bridges)
with the verified Multimodal Evidence Fusion layer.

Inputs:
  - Pre-event OSM: data/osm/dev/
  - Fused Evidence Vector: data/runs/fused_evidence_full_scene/fused_evidence_vector.geojson

Outputs written to:
  data/runs/infrastructure_impact_full_scene/
    1. infrastructure_impact.json
    2. affected_buildings.geojson
    3. affected_roads.geojson
    4. affected_bridges.geojson
    5. infrastructure_summary.json
    6. provenance.json
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# Ensure floodmap package is importable when invoked directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from floodmap.core.config import ImpactConfig, load_settings
from floodmap.core.contracts import DataMode
from floodmap.gis.infrastructure_analyzer import analyze_infrastructure_impact

DEFAULT_OSM_DIR = Path("data/osm/dev")
DEFAULT_EVIDENCE_VECTOR = Path("data/runs/fused_evidence_full_scene/fused_evidence_vector.geojson")
DEFAULT_OUT_DIR = Path("data/runs/infrastructure_impact_full_scene")
EVENT_DATE = date(2026, 8, 26)


def run_infrastructure_assessment(
    osm_dir: Path = DEFAULT_OSM_DIR,
    evidence_vector_path: Path = DEFAULT_EVIDENCE_VECTOR,
    out_dir: Path = DEFAULT_OUT_DIR,
    event_date: date = EVENT_DATE,
    target_crs: str = "EPSG:32645",
) -> None:
    print("=" * 80)
    print("STAGE 4: INFRASTRUCTURE IMPACT ASSESSMENT (TRISHULI SCENE)")
    print("=" * 80)
    print(f"OSM Pre-Event Source:    {osm_dir}")
    print(f"Fused Evidence Vector:   {evidence_vector_path}")
    print(f"Event Date:              {event_date.isoformat()}")
    print(f"Target Metric CRS:       {target_crs}")
    print(f"Output Directory:        {out_dir}")

    t0 = time.time()
    settings = load_settings()

    print("\n[1/4] Loading pre-event OSM and filtering to AOI bounds...")
    result = analyze_infrastructure_impact(
        osm_source=osm_dir,
        evidence_source=evidence_vector_path,
        event_date=event_date,
        target_crs=target_crs,
        config=settings.impact,
        analysis_id="infra_trishuli",
        data_mode=DataMode.REAL,
        aoi_name="trishuli_event_aoi",
    )

    print("\n[2/4] Saving Stage 4 output products...")
    saved_files = result.save_outputs(out_dir)
    for name, p in saved_files.items():
        print(f"  Written {name:25s} -> {p}")

    b_stats = result.building_stats
    r_stats = result.road_stats
    br_stats = result.bridge_stats

    print("\n[3/4] Infrastructure Impact Summary Metrics:")
    print("-" * 80)
    print("1. BUILDINGS:")
    print(f"   Total Buildings in Scene:      {b_stats.total_count:,d}")
    print(f"   Potentially Affected:         {b_stats.potentially_affected_count:,d} ({b_stats.affected_pct:.2f}%)")
    print(f"   Total Footprint Area:          {b_stats.total_footprint_m2:,.2f} m²")
    print(f"   Affected Footprint Area:       {b_stats.affected_footprint_m2:,.2f} m²")
    print(f"   By Dominant Evidence Tier:     {b_stats.by_evidence}")
    print(f"   By Intersecting Class:         {b_stats.by_class}")

    print("\n2. ROADS:")
    print(f"   Total Road Segments:           {r_stats.total_segments:,d} ({r_stats.total_length_km:.2f} km)")
    print(f"   Potentially Affected Segments: {r_stats.potentially_affected_segments:,d} ({r_stats.potentially_affected_length_km:.2f} km, {r_stats.affected_pct:.2f}%)")
    print(f"   By Dominant Evidence Tier:     {r_stats.by_evidence_km} km")
    print(f"   By Highway Classification:     {r_stats.by_highway_type_km}")

    print("\n3. BRIDGES:")
    print(f"   Total Bridges in Scene:        {br_stats.total_count:,d}")
    print(f"   Potentially Affected:         {br_stats.potentially_affected_count:,d} ({br_stats.affected_pct:.2f}%)")
    print(f"   By Dominant Evidence Tier:     {br_stats.by_evidence}")
    print(f"   By Intersecting Class:         {br_stats.by_class}")

    print("\n[4/4] Scientific Disclaimers & Integrity:")
    print(f"   * {result.impact_dict['scientific_disclaimer']}")
    print(f"   * Pre-event OSM snapshot: {result.impact_dict['osm_snapshot_date']} (Source: {result.impact_dict['osm_source']})")

    elapsed = time.time() - t0
    print(f"\nStage 4 completed in {elapsed:.2f} seconds.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 infrastructure impact assessment.")
    parser.add_argument("--osm-dir", type=Path, default=DEFAULT_OSM_DIR, help="Path to pre-event OSM directory")
    parser.add_argument("--evidence-vector", type=Path, default=DEFAULT_EVIDENCE_VECTOR, help="Path to fused evidence vector GeoJSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--target-crs", type=str, default="EPSG:32645", help="Target analysis CRS")
    args = parser.parse_args()

    run_infrastructure_assessment(
        osm_dir=args.osm_dir,
        evidence_vector_path=args.evidence_vector,
        out_dir=args.out_dir,
        target_crs=args.target_crs,
    )


if __name__ == "__main__":
    main()
