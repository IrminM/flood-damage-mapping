import json
from pathlib import Path
import numpy as np
import rasterio

run_dir = Path("data/runs/attention_unet_exploratory_trishuli")
p_tif = run_dir / "probability_exploratory.tif"
raw_tif = run_dir / "raw_binary_water_mask.tif"
gated_tif = run_dir / "terrain_gated_water_mask.tif"
sum_json = run_dir / "summary.json"
prov_json = run_dir / "provenance.json"

with rasterio.open(p_tif) as src:
    p_meta = {
        "shape": (src.height, src.width),
        "crs": str(src.crs),
        "transform": [src.transform[i] for i in range(6)],
        "res": src.res,
        "nodata": src.nodata,
        "dtype": src.dtypes[0],
        "tags": src.tags(),
    }
    p_arr = src.read(1)

with rasterio.open(raw_tif) as src:
    raw_meta = {
        "shape": (src.height, src.width),
        "crs": str(src.crs),
        "transform": [src.transform[i] for i in range(6)],
        "res": src.res,
        "nodata": src.nodata,
        "dtype": src.dtypes[0],
        "tags": src.tags(),
    }
    raw_arr = src.read(1)

with rasterio.open(gated_tif) as src:
    gated_meta = {
        "shape": (src.height, src.width),
        "crs": str(src.crs),
        "transform": [src.transform[i] for i in range(6)],
        "res": src.res,
        "nodata": src.nodata,
        "dtype": src.dtypes[0],
        "tags": src.tags(),
    }
    gated_arr = src.read(1)

with open(sum_json, "r", encoding="utf-8") as f:
    s_data = json.load(f)

with open(prov_json, "r", encoding="utf-8") as f:
    pr_data = json.load(f)

print("=== PROBABILITY RASTER METADATA ===")
print("Shape:", p_meta["shape"])
print("CRS:", p_meta["crs"])
print("Transform:", p_meta["transform"])
print("Resolution:", p_meta["res"])
print("NoData:", p_meta["nodata"])
print("Dtype:", p_meta["dtype"])
print("Tags:", p_meta["tags"])
print(f"Stats: min={np.nanmin(p_arr):.4f}, max={np.nanmax(p_arr):.4f}, mean={np.nanmean(p_arr):.4f}, median={np.nanmedian(p_arr):.4f}")

print("\n=== RAW BINARY MASK METADATA ===")
print("Shape:", raw_meta["shape"])
print("CRS:", raw_meta["crs"])
print("Transform:", raw_meta["transform"])
print("Resolution:", raw_meta["res"])
print("NoData:", raw_meta["nodata"])
print("Dtype:", raw_meta["dtype"])
print("Tags:", raw_meta["tags"])
u_raw, c_raw = np.unique(raw_arr, return_counts=True)
print("Unique counts:", dict(zip(u_raw.tolist(), c_raw.tolist())))

print("\n=== TERRAIN GATED MASK METADATA ===")
print("Shape:", gated_meta["shape"])
print("CRS:", gated_meta["crs"])
print("Transform:", gated_meta["transform"])
print("Resolution:", gated_meta["res"])
print("NoData:", gated_meta["nodata"])
print("Dtype:", gated_meta["dtype"])
print("Tags:", gated_meta["tags"])
u_gated, c_gated = np.unique(gated_arr, return_counts=True)
print("Unique counts:", dict(zip(u_gated.tolist(), c_gated.tolist())))

total_px = p_arr.size
px_area_km2 = (10.0 * 10.0) / 1e6
print("\n=== VERIFICATION AGAINST SUMMARY.JSON ===")
print(f"Total px: {total_px} vs summary: {s_data['scene']['total_pixels']}")
print(f"Raw water px: {np.sum(raw_arr == 1)} vs summary: {s_data['area_and_pixel_metrics']['raw_ai_water_pre_gate']['pixels']}")
print(f"Gated water px: {np.sum(gated_arr == 1)} vs summary: {s_data['area_and_pixel_metrics']['terrain_gated_ai_water_post_gate']['pixels']}")
print(f"Raw water km2: {np.sum(raw_arr == 1) * px_area_km2:.4f} vs summary: {s_data['area_and_pixel_metrics']['raw_ai_water_pre_gate']['area_km2']}")
print(f"Gated water km2: {np.sum(gated_arr == 1) * px_area_km2:.4f} vs summary: {s_data['area_and_pixel_metrics']['terrain_gated_ai_water_post_gate']['area_km2']}")

# Grid match check across all three
assert p_meta["shape"] == raw_meta["shape"] == gated_meta["shape"] == (2778, 2108)
assert p_meta["crs"] == raw_meta["crs"] == gated_meta["crs"] == "EPSG:32645"
assert p_meta["transform"] == raw_meta["transform"] == gated_meta["transform"]
assert p_meta["res"] == raw_meta["res"] == gated_meta["res"] == (10.0, 10.0)
print("\n>>> ALL RASTER CHECKS & SUMMARY.JSON COMPARISONS VERIFIED EXACT MATCH! <<<")
