#!/usr/bin/env python
"""Run a single real 256x256 patch smoke test using the trained Attention U-Net model.

Performs:
  1. Window extraction from co-registered S1 and S2 scenes.
  2. Strict channel ordering: [VV, VH, B1..B12].
  3. Preprocessing:
     - S1: float32, linear power (NO dB), per-channel finite median for NaN/Inf.
     - S2: float32, / 10000.0, clipped to [0, 1].
  4. Model inference via best_flood_attention_unet.keras.
  5. Thresholding at 0.40 to produce a binary flood/water mask.
  6. Saving output GeoTIFFs, numpy arrays, metadata, and summary to:
     data/runs/attention_unet_smoke_test/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

import _bootstrap  # noqa: F401

from floodmap.ai.adapters.attention_unet_adapter import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_INFERENCE_THRESHOLD,
    EXPECTED_15_CHANNELS,
    load_attention_unet_model,
    preprocess_s1,
    preprocess_s2,
)

S1_DEFAULT_PATH = Path("data/raw/dev/s1_after_20260828.tif")
S2_DEFAULT_PATH = Path("data/raw/dev/s2_after_20260827_l1c.tif")
RUN_DIR = Path("data/runs/attention_unet_smoke_test")


def run_smoke_test(
    s1_path: Path = S1_DEFAULT_PATH,
    s2_path: Path = S2_DEFAULT_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    out_dir: Path = RUN_DIR,
    row_start: int = 1200,
    col_start: int = 900,
    patch_size: int = 256,
    threshold: float = DEFAULT_INFERENCE_THRESHOLD,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ATTENTION U-NET: SINGLE REAL PATCH SMOKE TEST")
    print("=" * 60)

    # 1. Read patch window from S1 and S2
    win = Window(col_start, row_start, patch_size, patch_size)

    with rasterio.open(s1_path) as s1_src, rasterio.open(s2_path) as s2_src:
        print(f"Reading S1 patch ({s1_path.name}) at window {win}...")
        s1_raw = s1_src.read(window=win)  # (2, 256, 256)
        s1_crs = s1_src.crs
        s1_transform = window_transform(win, s1_src.transform)
        s1_patch_bounds = window_bounds(win, s1_src.transform)

        print(f"Reading S2 patch ({s2_path.name}) at window {win}...")
        s2_raw = s2_src.read(window=win)  # (13, 256, 256)
        s2_crs = s2_src.crs

        assert s1_crs == s2_crs, f"CRS mismatch: {s1_crs} != {s2_crs}"

    # 2. Input validation & preprocessing
    print("\nApplying proven preprocessing contract...")
    print("  S1: float32, linear power (NO dB), per-channel finite median NaN/Inf replacement")
    s1_pre = preprocess_s1(s1_raw)  # (2, 256, 256) float32
    assert s1_pre.shape == (2, patch_size, patch_size)
    assert np.isfinite(s1_pre).all(), "Non-finite values found in preprocessed S1"

    print("  S2: float32, divided by 10000.0, clipped to [0.0, 1.0]")
    s2_pre = preprocess_s2(s2_raw)  # (13, 256, 256) float32
    assert s2_pre.shape == (13, patch_size, patch_size)
    assert np.isfinite(s2_pre).all(), "Non-finite values found in preprocessed S2"
    assert s2_pre.min() >= 0.0 and s2_pre.max() <= 1.0, f"S2 out of [0, 1]: [{s2_pre.min()}, {s2_pre.max()}]"

    # 3. Assemble (H, W, 15) tensor
    s1_hwc = np.transpose(s1_pre, (1, 2, 0))  # (256, 256, 2)
    s2_hwc = np.transpose(s2_pre, (1, 2, 0))  # (256, 256, 13)
    patch_tensor = np.concatenate([s1_hwc, s2_hwc], axis=-1)  # (256, 256, 15)

    print(f"\nAssembled 15-channel patch shape: {patch_tensor.shape}")
    assert patch_tensor.shape == (patch_size, patch_size, 15)
    assert patch_tensor.dtype == np.float32

    # Add batch dimension: (1, 256, 256, 15)
    batch_input = np.expand_dims(patch_tensor, axis=0)

    # 4. Load real model
    print(f"\nLoading real Attention U-Net from {checkpoint_path}...")
    model = load_attention_unet_model(checkpoint_path)
    print(f"Model loaded successfully. Input shape: {model.input_shape}, Output shape: {model.output_shape}")

    # 5. Run inference on single patch
    print("\nRunning inference on single real patch...")
    pred_batch = model.predict(batch_input, verbose=0)  # (1, 256, 256, 1)
    prob_map = np.squeeze(pred_batch, axis=(0, -1)).astype(np.float32)  # (256, 256)

    assert prob_map.shape == (patch_size, patch_size)
    assert np.isfinite(prob_map).all(), "Model output contains non-finite values"

    prob_min = float(np.min(prob_map))
    prob_max = float(np.max(prob_map))
    prob_mean = float(np.mean(prob_map))

    print(f"Output probability statistics:")
    print(f"  Min:  {prob_min:.4f}")
    print(f"  Max:  {prob_max:.4f}")
    print(f"  Mean: {prob_mean:.4f}")

    # 6. Binary flood mask
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    flood_pixel_count = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    flood_pct = float(100.0 * flood_pixel_count / total_pixels)

    print(f"Decision threshold: {threshold:.2f}")
    print(f"  Pixels >= {threshold:.2f}: {flood_pixel_count} / {total_pixels} ({flood_pct:.2f}%)")

    # 7. Save outputs
    # A. Probability GeoTIFF & NPY
    prob_tif_path = out_dir / "probability_patch.tif"
    prob_npy_path = out_dir / "probability_patch.npy"
    np.save(prob_npy_path, prob_map)

    profile = {
        "driver": "GTiff",
        "height": patch_size,
        "width": patch_size,
        "count": 1,
        "dtype": "float32",
        "crs": s1_crs,
        "transform": s1_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(prob_tif_path, "w", **profile) as dst:
        dst.write(prob_map, 1)
        dst.set_band_description(1, "flood_probability")
        dst.update_tags(
            MODEL="Attention_UNet_S1_S2",
            CHECKPOINT=str(checkpoint_path),
            THRESHOLD=str(threshold),
            AREA_OR_POINT="Area",
        )

    # B. Binary Mask GeoTIFF & NPY
    mask_tif_path = out_dir / "flood_mask_patch.tif"
    mask_npy_path = out_dir / "flood_mask_patch.npy"
    np.save(mask_npy_path, binary_mask)

    mask_profile = profile.copy()
    mask_profile.update({"dtype": "uint8", "nodata": 255})
    with rasterio.open(mask_tif_path, "w", **mask_profile) as dst:
        dst.write(binary_mask, 1)
        dst.set_band_description(1, "binary_flood_mask")
        dst.update_tags(
            MODEL="Attention_UNet_S1_S2",
            THRESHOLD=str(threshold),
            AREA_OR_POINT="Area",
        )

    # C. Input Metadata JSON
    metadata = {
        "s1_source": str(s1_path),
        "s1_bands": ["VV", "VH"],
        "s2_source": str(s2_path),
        "s2_bands": [
            "B1", "B2", "B3", "B4", "B5", "B6",
            "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
        ],
        "channel_order": list(EXPECTED_15_CHANNELS),
        "input_tensor_shape": list(patch_tensor.shape),
        "window": {
            "row_start": row_start,
            "row_end": row_start + patch_size,
            "col_start": col_start,
            "col_end": col_start + patch_size,
        },
        "bounds": {
            "left": s1_patch_bounds[0],
            "bottom": s1_patch_bounds[1],
            "right": s1_patch_bounds[2],
            "top": s1_patch_bounds[3],
        },
        "crs": str(s1_crs),
        "transform": list(s1_transform)[:6],
        "preprocessing": {
            "s1": "float32, linear power, no dB conversion, per-channel finite median for NaN/Inf",
            "s2": "float32, divide by 10000.0, clip to [0.0, 1.0]",
        },
    }
    meta_path = out_dir / "input_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # D. Summary JSON
    summary = {
        "task": "Attention U-Net Single Real Patch Smoke Test",
        "checkpoint_path": str(checkpoint_path),
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "patch_location": {
            "row_start": row_start,
            "row_end": row_start + patch_size,
            "col_start": col_start,
            "col_end": col_start + patch_size,
            "utm_bounds": list(s1_patch_bounds),
            "crs": str(s1_crs),
        },
        "input_shape": list(patch_tensor.shape),
        "channel_order": list(EXPECTED_15_CHANNELS),
        "decision_threshold": threshold,
        "probability_stats": {
            "min": prob_min,
            "max": prob_max,
            "mean": prob_mean,
        },
        "classification_results": {
            "flood_water_pixels": flood_pixel_count,
            "total_pixels": total_pixels,
            "flood_water_percentage": flood_pct,
        },
        "files_created": [
            str(prob_tif_path),
            str(prob_npy_path),
            str(mask_tif_path),
            str(mask_npy_path),
            str(meta_path),
            str(out_dir / "summary.json"),
        ],
        "disclaimer": (
            "Technical smoke test only. Demonstrates end-to-end model ingestion and execution on real co-registered data. "
            "Does NOT constitute a validated flood damage assessment on the Himalayan Trishuli scene. "
            "Model does not classify debris, landslides, or structural building damage."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote results to {out_dir}:")
    print(f"  - {prob_tif_path.name}")
    print(f"  - {mask_tif_path.name}")
    print(f"  - {meta_path.name}")
    print(f"  - {summary_path.name}")
    print("=" * 60)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1", default=str(S1_DEFAULT_PATH))
    ap.add_argument("--s2", default=str(S2_DEFAULT_PATH))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--row-start", type=int, default=1200)
    ap.add_argument("--col-start", type=int, default=900)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=DEFAULT_INFERENCE_THRESHOLD)
    args = ap.parse_args()

    run_smoke_test(
        s1_path=Path(args.s1),
        s2_path=Path(args.s2),
        checkpoint_path=Path(args.checkpoint),
        out_dir=Path(args.out),
        row_start=args.row_start,
        col_start=args.col_start,
        patch_size=args.patch_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
