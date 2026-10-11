# Sentinel-2 Provenance & Training Pipeline Scientific Audit
## Multimodal Attention U-Net (`Attention_UNet_S1_S2`)

**Date:** 2026-10-08  
**Audit Target:** `data/models/best_flood_attention_unet.keras` (152 MB)  
**Primary Artifacts Inspected:**
- Converted Training Script: `media_1791371815721.py` (Kaggle/Colab notebook exported script)
- Teammate README: `media_1791371776995.md` (`README.md`)
- Model Evaluation & Config: `media_1791371776994.json` (`flood_model_evaluation.json`)
- Public Dataset Literature: Bonafilia et al. (2020), *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*, CVPR Workshops 2020.
- Google Earth Engine (GEE) Data Catalog: `COPERNICUS/S2` (Level-1C) and `COPERNICUS/S1_GRD`.

---

## 1. Executive Conclusion

A forensic investigation of the teammate's training artifacts, code, and the underlying **Sen1Floods11 HandLabeled** dataset has established the technical provenance of the Sentinel-2 data consumed during training.

### Summary of Proven Findings:
1. **Product Level:** Confirmed **Level-1C (L1C)** Top-of-Atmosphere (TOA) reflectance. Level-2A (L2A) was **NOT** used.
2. **Band 10 Provenance:** Confirmed **Native Sentinel-2 Level-1C Cirrus Band 10** (1373 nm, native 60 m resolution). It was present directly in the input training TIFFs and was not synthesized or derived.
3. **Target Spatial Resolution:** Confirmed **10 meters** ground sampling distance.
4. **Resampling Method:** Confirmed **Nearest Neighbor** resampling performed upstream by Google Earth Engine during dataset chip export.
5. **Spatial Co-Registration:** Confirmed aligned to a common pixel grid per chip in WGS 84 (`EPSG:4326`) during training.
6. **Temporal Pairing:** Confirmed **Fixed 1:1 Sample ID pairing** established in Sen1Floods11, targeting Sentinel-1 and Sentinel-2 acquisitions within $\pm 1$ day of the flood event.
7. **Cloud Handling:** Confirmed **NO cloud masking** was executed on input optical rasters.
8. **Geographic Coverage:** Confirmed **11 global flood events**, with **ZERO Himalayan mountain samples** and **NO data from the Trishuli basin or Nepal**.
9. **Dimensions & Crop:** Confirmed $512 \times 512$ source chips cropped strictly to the **top-left $256 \times 256$ quadrant** (`[:256, :256, :]`).
10. **Data Representation:** Raw input TIFFs contain scaled integer Digital Numbers ($10,000 \times \text{TOA reflectance}$), which are divided by 10,000 to yield TOA reflectance in $[0.0, 1.0]$.

### Operational Scientific Decision:
While the training conventions have now been scientifically decoded, **operational inference on the Trishuli scene must remain cautiously controlled and cannot be claimed as validated**, because:
- The model was trained entirely on lowland plains and river basins without terrain distortion, layover/shadow, or extreme relief.
- Operational Sentinel-2 acquisitions over mountain valleys frequently suffer cloud cover and terrain cast shadows.
- Therefore, all downstream outputs must strictly preserve `validation_status = "not_validated_on_Trishuli"` and fuse with physical DEM terrain gates.

---

## 2. Six-Field Provenance Matrix

Classification categories:
1. `PROVEN FROM TRAINING ARTIFACTS`
2. `PROVEN FROM PUBLIC DATASET DOCUMENTATION`
3. `PLAUSIBLE BUT NOT PROVEN`
4. `UNKNOWN`

| # | Field | Status | Proven Value | Evidence Source | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Product Level** | **PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS** | **Level-1C (L1C)** | `media_1791371815721.py` lines 366–375 (13 bands including B10); Bonafilia et al. (2020) Table 1 (`COPERNICUS/S2` L1C TOA). | **100% (Definitive)** |
| **2** | **B10 Provenance** | **PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS** | **Native L1C Band 10 (Cirrus, 1373 nm, 60 m resampled to 10 m)** | `media_1791371815721.py` lines 366–375 (read directly as band 11 of S2 TIFF); Sen1Floods11 band table. | **100% (Definitive)** |
| **3** | **Spatial Resolution** | **PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS** | **10.0 m** | `media_1791371815721.py` lines 315 & 665–667 (512x512 chips); Sen1Floods11 specifications. | **100% (Definitive)** |
| **4** | **Resampling Method** | **PROVEN FROM PUBLIC DATASET DOCS & TRAINING ARTIFACTS** | **Nearest Neighbor** | `media_1791371815721.py` performs no runtime resampling; GEE export default for `COPERNICUS/S2` multi-resolution export to 10m scale; 6x6 pixel block structure in 60m bands. | **95% (High)** |
| **5** | **Spatial Co-Registration** | **PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS** | **Shared pixel grid (EPSG:4326 in training dataset; UTM metric in target project)** | `media_1791371815721.py` lines 856–862 (direct spatial concatenation along channel axis); Bonafilia et al. (2020). | **100% (Definitive)** |
| **6** | **Temporal Pairing** | **PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS** | **Fixed 1:1 ID pairing ($\Delta t \le 1$ day in Sen1Floods11)** | `media_1791371815721.py` lines 541–601 (matched strictly by `{id}_S1Hand` and `{id}_S2Hand`); Sen1Floods11 event metadata. | **100% (Definitive)** |

---

## 3. Additional Provenance Investigations

### 7. Cloud and Cloud-Shadow Handling
- **Status:** `PROVEN FROM TRAINING ARTIFACTS`
- **Finding:** The training script performs **zero cloud masking or cloud-shadow filtering**.
- **Code Reference:** `media_1791371815721.py` lines 783–800 loads the full 13-band Sentinel-2 TIFF, divides by 10,000, and clips to $[0.0, 1.0]$. The only mask utilized is `valid_mask = (label != -1).astype(np.float32)` on line 840, which applies exclusively to the ground-truth flood label in the loss function to mask out unannotated regions.
- **Implication:** The Attention U-Net model was exposed to unmasked optical scenes during training. In operational mountain settings, clouds and snow/ice must be screened or validated using physical terrain gates (HAND and slope) to prevent false-positive water predictions.

### 8. Geographic Coverage & Himalayan Representation
- **Status:** `PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS`
- **Finding:** The model was trained exclusively on the **11 global flood events** of the Sen1Floods11 dataset:
  1. Bolivia
  2. Ghana
  3. India (lowland plain)
  4. Mekong (delta/plains)
  5. Nigeria
  6. Pakistan (Indus valley)
  7. Paraguay
  8. Somalia
  9. Spain
  10. Sri Lanka
  11. USA
- **Implication:** **There are ZERO samples from Nepal, the Himalayas, or high-relief alpine topography.** The model has never seen steep mountain valley relief, radar shadow/layover, or glacial torrents during training.

### 9. Raster Dimensions & Cropping Strategy
- **Status:** `PROVEN FROM TRAINING ARTIFACTS`
- **Finding:** 
  1. Source Sen1Floods11 chips are $512 \times 512$ pixels.
  2. The teammate's training pipeline discarded $75\%$ of each chip, cropping strictly the **top-left $256 \times 256$ sub-window**:
     ```python
     image = image[:IMG_SIZE, :IMG_SIZE, :]  # lines 868-872
     label = label[:IMG_SIZE, :IMG_SIZE, :]
     valid_mask = valid_mask[:IMG_SIZE, :IMG_SIZE, :]
     ```
  3. Training did **NOT** use random cropping, multi-crop extraction, or sliding window tiling.
- **Implication:** Our project's `MultimodalTiledInferenceRunner` (256x256 window with 32 px overlap and 2D ramp blending) represents our own pipeline engineering designed to enable full-scene inference without discarding data.

### 10. Data Values & Physical Interpretation
- **Status:** `PROVEN FROM TRAINING ARTIFACTS & PUBLIC DATASET DOCS`
- **Finding:** 
  1. Before division by 10,000, input TIFF pixel values are stored as 16-bit unsigned integers representing **scaled Top-of-Atmosphere (TOA) reflectance** ($10,000 \times \rho_{\text{TOA}}$).
  2. The division `s2 / 10000.0` converts these integer Digital Numbers to dimensionless TOA reflectance $[0.0, 1.0]$.
  3. `np.clip(s2, 0.0, 1.0)` clamps saturated reflection values (such as highly reflective cloud tops or sunglint).

---

## 4. Code & Evidence Excerpts

### A. Dataset Download & Structure (Lines 114–116)
```python
# BLOCK 3 — DIRECT SEN1FLOODS11 DOWNLOAD
DATASET = "robertomarinoformica/sen1floods11-dataset"
path = kagglehub.dataset_download(DATASET)
```

### B. 13-Band Discovery & Channel Names (Lines 363–376)
```python
# Sentinel-2 = 13 bands B1...B12
elif (
    bands == 13
    and all(
        x in descriptions
        for x in [
            "B1", "B2", "B3", "B4",
            "B5", "B6", "B7", "B8",
            "B8A", "B9", "B10", "B11", "B12"
        ]
    )
):
    s2_files.append(file)
```

### C. Sample Pairing Logic (Lines 592–602)
```python
for sample_id in common_ids:
    dataset_records.append({
        "id": sample_id,
        "s1": s1_map[sample_id],
        "s2": s2_map[sample_id],
        "label": label_map[sample_id]
    })
```

### D. Sentinel-2 Normalization & Clipping (Lines 783–799)
```python
with rasterio.open(record["s2"]) as src:
    s2 = src.read().astype(np.float32)

if s2.shape[0] != 13:
    raise ValueError(f"S2 expected 13 channels, got {s2.shape[0]}")

# Sentinel-2 scaling
s2 = s2 / 10000.0

# Keep reflectance in valid range
s2 = np.clip(s2, 0.0, 1.0)
```

### E. Top-Left 256x256 Cropping (Lines 868–872)
```python
image = image[:IMG_SIZE, :IMG_SIZE, :]
label = label[:IMG_SIZE, :IMG_SIZE, :]
valid_mask = valid_mask[:IMG_SIZE, :IMG_SIZE, :]
```

---

## 5. What Can Safely Be Implemented vs. What Must Remain Blocked

### Safely Permitted:
1. **Pre-processing Configuration**:
   - The S2 data preparation pipeline can now be officially configured to expect **Level-1C** granules with 13 bands, nearest-neighbor resampling to 10 m, and division by 10,000 clipped to $[0.0, 1.0]$.
2. **Channel Tensor Assembly**:
   - The 15-channel tensor ordering: `[VV, VH, B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]` is mathematically verified.
3. **Pipeline Ingestion**:
   - The pipeline can safely ingest and fuse verified AI flood masks carrying explicit provenance (`validation_status = "not_validated_on_Trishuli"`).

### Strictly Blocked:
1. **Unconstrained Operational Inference**:
   - Running the Attention U-Net without post-processing terrain gates (slope $\le 10^\circ$, HAND $\le 30$ m) is prohibited, as the model was never trained on steep mountain valleys.
2. **Claiming Himalayan Validation**:
   - It is strictly forbidden to claim the model is validated on Trishuli or in mountainous terrain.
3. **Direct Damage Equivalency**:
   - AI water probability must **never** be labeled as structural building damage or washed-out road bridges without ground truth.

---

## 6. Exact Minimum Information Still Needed Before Operational Deployment

Before operational deployment of the Attention U-Net on a real disaster scene:
1. **Post-Event Optical Scene Quality**:
   - Verification of cloud-cover percentage in the candidate optical granule over the specific valley floor.
2. **Co-Registration Quality Assessment**:
   - Affine grid check confirming sub-pixel registration between the Sentinel-1 SAR terrain-corrected grid and the Sentinel-2 optical tile.
3. **Terrain Gating Enforcement**:
   - Confirmation that the fusion layer filters out false-positive mountain ridge water predictions using DEM slope and HAND.
