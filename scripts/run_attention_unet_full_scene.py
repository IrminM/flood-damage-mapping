#!/usr/bin/env python
"""Run full-scene tiled inference over the entire Trishuli scene using the Attention U-Net model.

Inputs:
  - Sentinel-1 Post-Event: data/raw/dev/s1_after_20260828.tif
  - Sentinel-2 Post-Event: data/raw/dev/s2_after_20260827_l1c.tif
  - Model Checkpoint: data/models/best_flood_attention_unet.keras

Inference Configuration:
  - Tile size: 256x256
  - Overlap: 32 px
  - Stride: 224 px
  - 2D ramp blending over overlap margins
  - Decision threshold: 0.40

Outputs in data/runs/attention_unet_full_scene/:
  - probability_full_scene.tif
  - flood_water_mask_full_scene.tif
  - summary.json
  - provenance.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import keras
import numpy as np
import rasterio

import _bootstrap  # noqa: F401

from floodmap.ai.adapters.attention_unet_adapter import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_INFERENCE_THRESHOLD,
    EXPECTED_15_CHANNELS,
    assemble_full_scene_tensor,
    load_attention_unet_model,
)
from floodmap.ai.inference import MultimodalTiledInferenceRunner
from floodmap.core.georaster import GeoRaster

S1_PATH = Path("data/raw/dev/s1_after_20260828.tif")
S2_PATH = Path("data/raw/dev/s2_after_20260827_l1c.tif")
OUT_DIR = Path("data/runs/attention_unet_full_scene")


def run_full_scene_inference(
    s1_path: Path = S1_PATH,
    s2_path: Path = S2_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    out_dir: Path = OUT_DIR,
    threshold: float = DEFAULT_INFERENCE_THRESHOLD,
    tile_size: int = 256,
    overlap: int = 32,
) -> dict:
    start_time = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ATTENTION U-NET: FULL-SCENE TILED INFERENCE (TRISHULI)")
    print("=" * 70)
    print(f"S1 Post-Event Source: {s1_path}")
    print(f"S2 Post-Event Source: {s2_path}")
    print(f"Model Checkpoint:     {checkpoint_path}")
    print(f"Tile Size / Overlap:  {tile_size}x{tile_size} px / {overlap} px overlap (stride = {tile_size - overlap} px)")
    print(f"Decision Threshold:   {threshold:.2f}")

    # 1. Load S1 and S2 scenes
    print("\n[1/6] Loading source rasters and verifying spatial reference...")
    with rasterio.open(s1_path) as s1_src, rasterio.open(s2_path) as s2_src:
        s1_data = s1_src.read()  # (2, 2778, 2108)
        s2_data = s2_src.read()  # (13, 2778, 2108)
        ref_crs = s1_src.crs
        ref_transform = s1_src.transform
        ref_bounds = s1_src.bounds
        ref_height = s1_src.height
        ref_width = s1_src.width

        assert s1_src.crs == s2_src.crs, f"CRS mismatch: {s1_src.crs} != {s2_src.crs}"
        assert s1_src.shape == s2_src.shape, f"Shape mismatch: {s1_src.shape} != {s2_src.shape}"
        assert s1_src.transform == s2_src.transform, f"Transform mismatch: {s1_src.transform} != {s2_src.transform}"
        assert s1_src.bounds == s2_src.bounds, f"Bounds mismatch: {s1_src.bounds} != {s2_src.bounds}"
        assert s1_data.shape[0] == 2, f"Expected 2 S1 bands, got {s1_data.shape[0]}"
        assert s2_data.shape[0] == 13, f"Expected 13 S2 bands, got {s2_data.shape[0]}"

    print(f"  Scene Dimensions: {ref_height} rows x {ref_width} cols ({ref_height * ref_width:,} pixels)")
    print(f"  Spatial CRS:      {ref_crs}")
    print(f"  Affine Transform: {ref_transform}")

    # 2. Assemble 15-channel scene tensor with validated preprocessing
    print("\n[2/6] Assembling full scene 15-channel tensor (channels-last)...")
    print("  Applying proven preprocessing:")
    print("    S1: float32 linear power, per-channel finite median replacement for NaN/Inf")
    print("    S2: float32 / 10000.0, clipped to [0.0, 1.0]")
    
    full_scene_hwc = assemble_full_scene_tensor(
        s1_data=s1_data,
        s2_data=s2_data,
    )
    assert full_scene_hwc.shape == (ref_height, ref_width, 15)
    assert full_scene_hwc.dtype == np.float32
    assert np.isfinite(full_scene_hwc).all(), "Non-finite values detected in preprocessed tensor"
    print(f"  Assembled Tensor Shape: {full_scene_hwc.shape} ({full_scene_hwc.nbytes / 1e6:.1f} MB)")

    # 3. Load model
    print(f"\n[3/6] Loading Attention U-Net model from {checkpoint_path}...")
    model = load_attention_unet_model(checkpoint_path)
    print(f"  Loaded model successfully. Input: {model.input_shape}, Output: {model.output_shape}")

    # 4. Run MultimodalTiledInferenceRunner
    print(f"\n[4/6] Executing tiled sliding-window inference with 2D ramp blending...")
    runner = MultimodalTiledInferenceRunner(
        tile_size=tile_size,
        overlap=overlap,
        blend_mode="ramp",
    )

    stride = tile_size - overlap
    H, W, C = full_scene_hwc.shape
    pad_h = max(0, tile_size - H) if H < tile_size else (stride - (H - tile_size) % stride) % stride
    pad_w = max(0, tile_size - W) if W < tile_size else (stride - (W - tile_size) % stride) % stride
    p_H = H + pad_h
    p_W = W + pad_w
    y_steps = list(range(0, p_H - tile_size + 1, stride))
    x_steps = list(range(0, p_W - tile_size + 1, stride))
    expected_tile_count = len(y_steps) * len(x_steps)
    print(f"  Tile Grid: {len(y_steps)} rows x {len(x_steps)} cols = {expected_tile_count} total tiles")
    print(f"  Padded canvas dimensions: ({p_H}, {p_W})")

    tile_counter = 0
    t0 = time.time()

    def _predict_with_progress(batch_arr: np.ndarray) -> np.ndarray:
        nonlocal tile_counter
        preds = model.predict(batch_arr, verbose=0)
        tile_counter += batch_arr.shape[0]
        if tile_counter % 20 == 0 or tile_counter == expected_tile_count:
            elapsed = time.time() - t0
            print(f"    Processed tile {tile_counter:3d}/{expected_tile_count} ({100.0 * tile_counter / expected_tile_count:.1f}%) "
                  f"[{elapsed:.1f}s, {elapsed / max(tile_counter, 1):.2f}s/tile]")
        return preds

    ref_georaster = GeoRaster(
        data=np.zeros((1, ref_height, ref_width), dtype=np.float32),
        transform=(
            ref_transform.a,
            ref_transform.b,
            ref_transform.c,
            ref_transform.d,
            ref_transform.e,
            ref_transform.f,
        ),
        crs=str(ref_crs),
        band_names=["water_prob"],
    )

    prob_raster = runner.predict(
        predict_fn=_predict_with_progress,
        scene_hwc=full_scene_hwc,
        reference_raster=ref_georaster,
    )

    assert tile_counter == expected_tile_count, f"Processed {tile_counter} tiles, expected {expected_tile_count}"
    prob_array = prob_raster.data[0]  # (2778, 2108) float32
    assert prob_array.shape == (ref_height, ref_width), f"Output shape mismatch: {prob_array.shape}"
    assert np.isfinite(prob_array).all(), "Non-finite values present in full-scene probability map"

    # 5. Thresholding & Statistics
    print("\n[5/6] Computing probability statistics and binary flood mask...")
    prob_min = float(np.min(prob_array))
    prob_max = float(np.max(prob_array))
    prob_mean = float(np.mean(prob_array))
    prob_median = float(np.median(prob_array))

    binary_mask = (prob_array >= threshold).astype(np.uint8)
    total_pixels = int(prob_array.size)
    flood_pixels = int(np.sum(binary_mask == 1))
    background_pixels = int(np.sum(binary_mask == 0))
    flood_pct = float(100.0 * flood_pixels / total_pixels)
    bg_pct = float(100.0 * background_pixels / total_pixels)

    pixel_area_m2 = 10.0 * 10.0  # 100 m²
    total_area_km2 = float(total_pixels * pixel_area_m2 / 1e6)
    flood_area_km2 = float(flood_pixels * pixel_area_m2 / 1e6)

    print(f"  Probability Range:  [{prob_min:.4f}, {prob_max:.4f}] (mean: {prob_mean:.4f}, median: {prob_median:.4f})")
    print(f"  Total Scene Area:   {total_area_km2:.4f} km² ({total_pixels:,} pixels)")
    print(f"  Flood/Water Pixels: {flood_pixels:,} ({flood_pct:.2f}%) -> {flood_area_km2:.4f} km²")
    print(f"  Background Pixels:  {background_pixels:,} ({bg_pct:.2f}%)")

    # 6. Write GeoTIFF outputs, summary, and provenance
    print("\n[6/6] Writing full-scene GeoTIFF outputs, arrays, and metadata...")
    prob_tif = out_dir / "probability_full_scene.tif"
    mask_tif = out_dir / "flood_water_mask_full_scene.tif"
    prob_npy = out_dir / "probability_full_scene.npy"
    mask_npy = out_dir / "flood_water_mask_full_scene.npy"
    summary_path = out_dir / "summary.json"
    prov_path = out_dir / "provenance.json"

    # Save NumPy arrays
    np.save(prob_npy, prob_array)
    np.save(mask_npy, binary_mask)

    # Save Probability GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": 1,
        "dtype": "float32",
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(prob_tif, "w", **profile) as dst:
        dst.write(prob_array, 1)
        dst.set_band_description(1, "ai_flood_water_probability")
        dst.update_tags(
            MODEL="Attention_UNet_S1_S2",
            CHECKPOINT=str(checkpoint_path),
            THRESHOLD=str(threshold),
            TILE_SIZE=str(tile_size),
            OVERLAP=str(overlap),
            BLEND_MODE="ramp",
            AREA_OR_POINT="Area",
            DESCRIPTION="Full-scene AI flood water probability (unvalidated on Himalayan terrain)",
        )

    # Save Binary Mask GeoTIFF
    mask_profile = profile.copy()
    mask_profile.update({"dtype": "uint8", "nodata": 255})
    with rasterio.open(mask_tif, "w", **mask_profile) as dst:
        dst.write(binary_mask, 1)
        dst.set_band_description(1, "ai_flood_water_mask")
        dst.update_tags(
            MODEL="Attention_UNet_S1_S2",
            THRESHOLD=str(threshold),
            AREA_OR_POINT="Area",
            CLASS_0="Background",
            CLASS_1="AI Flood/Water Candidate",
            DESCRIPTION="Binary water/flood mask at threshold 0.40 (does NOT classify debris or building damage)",
        )

    # Provenance JSON
    provenance = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "data_mode": "REAL",
        "inputs": {
            "s1_after": {
                "path": str(s1_path),
                "bands": ["VV", "VH"],
                "acquisition": "2026-08-28T12:21:41.762Z",
                "units": "linear",
            },
            "s2_after": {
                "path": str(s2_path),
                "bands": list(EXPECTED_15_CHANNELS[2:]),
                "acquisition": "2026-08-27T04:56:59.024Z",
                "units": "reflectance",
                "cloud_cover_pct": 65.6,
            },
            "channel_contract": list(EXPECTED_15_CHANNELS),
        },
        "model": {
            "architecture": "Attention_UNet_S1_S2",
            "framework": "TensorFlow / Keras 3",
            "checkpoint_path": str(checkpoint_path),
            "input_shape": [None, 256, 256, 15],
            "output_shape": [None, 256, 256, 1],
            "decision_threshold": threshold,
            "training_dataset": "Sen1Floods11 HandLabeled",
        },
        "tiling": {
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": stride,
            "blend_mode": "ramp",
            "total_tiles_processed": expected_tile_count,
            "grid_dimensions": {"rows": len(y_steps), "cols": len(x_steps)},
            "padded_canvas": [p_H, p_W],
        },
        "spatial_reference": {
            "crs": str(ref_crs),
            "dimensions": [ref_height, ref_width],
            "resolution_m": [10.0, 10.0],
            "bounds": [ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top],
            "transform": list(ref_transform)[:6],
        },
    }
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Summary JSON
    summary = {
        "task": "Attention U-Net Full-Scene Tiled Inference (Trishuli Post-Event)",
        "status": "COMPLETED",
        "total_tiles_processed": expected_tile_count,
        "scene_dimensions": [ref_height, ref_width],
        "total_scene_area_km2": round(total_area_km2, 4),
        "decision_threshold": threshold,
        "probability_statistics": {
            "min": round(prob_min, 4),
            "max": round(prob_max, 4),
            "mean": round(prob_mean, 4),
            "median": round(prob_median, 4),
        },
        "ai_segmentation_results": {
            "flood_water_pixels": flood_pixels,
            "flood_water_percentage": round(flood_pct, 4),
            "flood_water_area_km2": round(flood_area_km2, 4),
            "background_pixels": background_pixels,
            "background_percentage": round(bg_pct, 4),
            "total_pixels": total_pixels,
        },
        "spatial_validation": {
            "crs": str(ref_crs),
            "pixel_resolution_m": 10.0,
            "transform_matches_s1": True,
            "dimensions_match_s1": True,
            "no_pixels_omitted": True,
        },
        "files_created": [
            str(prob_tif),
            str(mask_tif),
            str(prob_npy),
            str(mask_npy),
            str(summary_path),
            str(prov_path),
        ],
        "scientific_disclaimer": (
            "This output represents AI flood/water segmentation from the teammate Attention U-Net model. "
            "It is NOT a validated flood-damage map. The model predicts open water/flood probability only, "
            "and does NOT classify debris flows, landslides, mud accumulation, or building structural damage. "
            "The model was trained on Sen1Floods11 and has not been validated on steep Himalayan mountain terrain. "
            "Stage 2 SAR scientific change detection and terrain gates remain separate and unaltered."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    total_elapsed = time.time() - start_time
    print(f"\nFull-scene inference complete in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)!")
    print(f"Results written to: {out_dir}")
    print("=" * 70)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1", default=str(S1_PATH))
    ap.add_argument("--s2", default=str(S2_PATH))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--threshold", type=float, default=DEFAULT_INFERENCE_THRESHOLD)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args()

    run_full_scene_inference(
        s1_path=Path(args.s1),
        s2_path=Path(args.s2),
        checkpoint_path=Path(args.checkpoint),
        out_dir=Path(args.out),
        threshold=args.threshold,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
