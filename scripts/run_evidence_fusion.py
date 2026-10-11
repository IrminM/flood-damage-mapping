"""Execute Multimodal Evidence Fusion on Trishuli Full Scene.

Fuses:
  - Attention U-Net Full-Scene Inundation Probability (data/runs/attention_unet_full_scene/)
  - Stage 2 Sentinel-1 Baseline Change Detection (data/runs/a5cafa318f0c/change/)
  - Physical Copernicus DEM GLO-30 Terrain Gates (data/runs/a5cafa318f0c/terrain/)

Writes results to:
  data/runs/fused_evidence_full_scene/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure floodmap package is importable when invoked directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from floodmap.fusion.evidence_fusion import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_NAMES,
    STRENGTH_DESCRIPTIONS,
    STRENGTH_NAMES,
    EvidenceCategory,
    fuse_multimodal_evidence,
    write_fusion_outputs,
)
from floodmap.io.geotiff import read_geotiff

DEFAULT_AI_PROB = Path("data/runs/attention_unet_full_scene/probability_full_scene.tif")
DEFAULT_AI_MASK = Path("data/runs/attention_unet_full_scene/flood_water_mask_full_scene.tif")
DEFAULT_S2_RUN_DIR = Path("data/runs/a5cafa318f0c")
DEFAULT_OUT_DIR = Path("data/runs/fused_evidence_full_scene")


def run_fusion(
    ai_prob_path: Path = DEFAULT_AI_PROB,
    ai_mask_path: Path = DEFAULT_AI_MASK,
    s2_run_dir: Path = DEFAULT_S2_RUN_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    ai_threshold: float = 0.40,
    water_max_slope_deg: float = 10.0,
    water_max_hand_m: float = 30.0,
    change_max_slope_deg: float = 25.0,
) -> None:
    print("=" * 75)
    print("MULTIMODAL EVIDENCE FUSION: TRISHULI POST-EVENT SCENE")
    print("=" * 75)
    print(f"AI Probability Source:    {ai_prob_path}")
    print(f"AI Mask Source:           {ai_mask_path}")
    print(f"Stage 2 SAR Source Dir:   {s2_run_dir}")
    print(f"Output Directory:         {out_dir}")
    print(f"AI Water Threshold:       {ai_threshold:.2f}")
    print(f"Terrain Slope Gate:       <= {water_max_slope_deg:.1f} deg (water), <= {change_max_slope_deg:.1f} deg (change)")
    print(f"Terrain HAND Gate:        <= {water_max_hand_m:.1f} m")

    t0 = time.time()

    # 1. Load AI products
    print("\n[1/5] Loading AI segmentation rasters...")
    ai_prob = read_geotiff(ai_prob_path)
    ai_mask = read_geotiff(ai_mask_path)
    print(f"  AI Grid: {ai_prob.shape} | CRS: {ai_prob.crs} | Transform: {ai_prob.transform}")

    # 2. Load Stage 2 SAR & Terrain products
    print("\n[2/5] Loading Stage 2 SAR change and physical terrain products...")
    s2_cls = read_geotiff(s2_run_dir / "change/change_classes.tif")
    s2_excl = read_geotiff(s2_run_dir / "change/exclusion_flags.tif")
    s2_ctx = read_geotiff(s2_run_dir / "change/change_context.tif")
    s2_slope = read_geotiff(s2_run_dir / "terrain/slope_deg.tif")
    s2_hand = read_geotiff(s2_run_dir / "terrain/hand_m.tif")
    s2_tflags = read_geotiff(s2_run_dir / "terrain/terrain_flags.tif")
    print(f"  Stage 2 Grid: {s2_cls.shape} | CRS: {s2_cls.crs}")

    # 3. Assemble source metadata for provenance
    source_meta = {
        "ai_probability_path": str(ai_prob_path),
        "ai_mask_path": str(ai_mask_path),
        "ai_model": "Attention_UNet_S1_S2",
        "ai_threshold": ai_threshold,
        "stage2_run_dir": str(s2_run_dir),
        "stage2_method": "s1_logratio_v1",
        "stage2_inputs": {
            "before_scene": "data/raw/dev/s1_before_20260816.tif",
            "after_scene": "data/raw/dev/s1_after_20260828.tif",
            "dem": "data/dem/dev/dem_glo30.tif",
        },
    }

    # 4. Execute fusion
    print("\n[3/5] Fusing multimodal evidence and applying physical terrain gates...")
    fusion_result = fuse_multimodal_evidence(
        ai_probability=ai_prob,
        stage2_classes=s2_cls,
        ai_mask=ai_mask,
        stage2_exclusion=s2_excl,
        stage2_context=s2_ctx,
        terrain_slope=s2_slope,
        terrain_hand=s2_hand,
        terrain_flags=s2_tflags,
        ai_threshold=ai_threshold,
        water_max_slope_deg=water_max_slope_deg,
        water_max_hand_m=water_max_hand_m,
        change_max_slope_deg=change_max_slope_deg,
        source_metadata=source_meta,
    )

    # 5. Write outputs
    print("\n[4/5] Writing full-scene GeoTIFF, GeoJSON, summary, and provenance...")
    outputs = write_fusion_outputs(fusion_result, out_dir, write_vector=True)
    for k, p in outputs.items():
        print(f"  Written {k:22s} -> {p}")

    # 6. Display Summary Table
    print("\n[5/5] Multimodal Evidence Statistics:")
    print("-" * 80)
    print(f"{'Category':<32} {'Code':<5} {'Pixels':>10} {'Area (km²)':>12} {'% Obs':>8}")
    print("-" * 80)
    summary = fusion_result.summary
    for cat_name, stats in summary["evidence_categories"].items():
        code = stats["code"]
        px = stats["pixels"]
        km2 = stats["area_km2"]
        pct_obs = f"{stats['pct_observable']:.2f}%" if code != 255 else "N/A"
        print(f"{cat_name:<32} {code:<5} {px:>10,d} {km2:>12.4f} {pct_obs:>8}")
    print("-" * 80)
    print(f"Total Observable Area:  {summary['spatial_reference']['observable_area_km2']:.4f} km² ({summary['spatial_reference']['observable_pixels']:,} px)")
    print(f"Total AOI Scene Area:   {summary['spatial_reference']['total_scene_area_km2']:.4f} km² ({summary['spatial_reference']['total_pixels']:,} px)")

    print("\nHigh-Level Scientific Groupings:")
    sg = summary["scientific_groupings"]
    print(f"  1. Dual-Evidence Valley Change (AI + SAR): {sg['possible_event_surface_change_km2']['dual_evidence_valley_change']:.4f} km²")
    print(f"  2. SAR-Only Valley Change:                 {sg['possible_event_surface_change_km2']['sar_only_valley_change']:.4f} km²")
    print(f"  3. AI-Only Water Candidate (Lowland):      {sg['flood_open_water_km2']['ai_only_water_candidate']:.4f} km²")
    print(f"  4. Permanent Water Bodies:                 {sg['flood_open_water_km2']['permanent_standing_water']:.4f} km²")
    print(f"  5. SAR Hillside Change (Unvalidated):      {sg['unvalidated_change_km2']['sar_hillside_change']:.4f} km²")
    print(f"  6. Terrain-Rejected Candidates:            {sg['unvalidated_change_km2']['terrain_rejected_candidates']:.4f} km²")
    print(f"  7. Observable Background (No Change):      {sg['observable_background_km2']:.4f} km²")
    print(f"  8. Outside AOI / No Data:                  {sg['invalid_or_nodata_km2']:.4f} km²")

    elapsed = time.time() - t0
    print(f"\nFusion complete in {elapsed:.2f} seconds.")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal evidence fusion on Trishuli scene.")
    parser.add_argument("--ai-prob", type=Path, default=DEFAULT_AI_PROB, help="Path to AI probability GeoTIFF")
    parser.add_argument("--ai-mask", type=Path, default=DEFAULT_AI_MASK, help="Path to AI binary mask GeoTIFF")
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_RUN_DIR, help="Path to Stage 2 run directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--ai-threshold", type=float, default=0.40, help="AI probability decision threshold")
    parser.add_argument("--water-slope", type=float, default=10.0, help="Max slope for water (deg)")
    parser.add_argument("--water-hand", type=float, default=30.0, help="Max HAND for water (m)")
    parser.add_argument("--change-slope", type=float, default=25.0, help="Max slope for change (deg)")
    args = parser.parse_args()

    run_fusion(
        ai_prob_path=args.ai_prob,
        ai_mask_path=args.ai_mask,
        s2_run_dir=args.s2_dir,
        out_dir=args.out_dir,
        ai_threshold=args.ai_threshold,
        water_max_slope_deg=args.water_slope,
        water_max_hand_m=args.water_hand,
        change_max_slope_deg=args.change_slope,
    )


if __name__ == "__main__":
    main()
