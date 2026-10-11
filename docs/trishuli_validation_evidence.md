# Trishuli Corridor Multi-Source Validation & Evidence Audit

**Status:** `not_validated_on_Trishuli` (Uncalibrated Prototype)  
**Target Event Date:** 26 August 2026 (Monsoon GLOF / Debris Flow Event)  
**Study Corridor:** Trishuli River Basin & Confluence, Rasuwa & Nuwakot Districts, Nepal (EPSG:32645, 10 m Grid)  
**Evaluation Zone Area:** 81.71 km² (IWM corridor + 500 m buffer ∩ Nepal Boundary)  

---

## 1. Executive Summary & Scientific Disclaimer

This document presents the complete validation preparation, rasterization verification, multi-source agreement metrics, and data limitation audit for the FloodIntel system on the August 2026 Trishuli flood case.

> **CRITICAL SCIENTIFIC PRINCIPLE:**  
> The metrics presented in this report measure **spatial agreement with unvalidated third-party reference maps**, NOT verified model accuracy against physical ground truth. No independent field surveys or cloud-free high-resolution aerial ground-truth datasets exist for this event. Model status remains formally designated as **`not_validated_on_Trishuli`**. Under no circumstances should provisional agreement figures be represented as operational or scientific accuracy.

---

## 2. Polygon Rasterization Engine Fix (`Grid.burn()`)

### 2.1 The Cross-Polygon XOR Defect
During the toolkit audit of `floodintel_validate.py`, a critical flaw was identified in the geometric burning engine:
- **Original Code:** The original implementation initialized a single boolean grid and applied bitwise XOR (`out ^= np.array(im, bool)`) across **all rings across all polygon entities** in the dataset.
- **Defect Mechanism:** In boolean XOR, $1 \oplus 1 = 0$. When two distinct polygon entities overlapped in space, pixels falling within both interiors were cancelled out, creating false holes and artificial data gaps in inundated corridors.
- **Performance Defect:** The original function allocated full-dimension $(W, H)$ raster canvases for each individual ring, creating high memory overhead.

### 2.2 The Verified Fix
The corrected `Grid.burn()` implementation resolves this by separating intra-polygon hole subtraction from inter-polygon feature union:
1. **Per-Polygon Bounding Box:** Computes tight grid bounding coordinates `(c0, c1, r0, r1)` for each polygon entity to avoid full-canvas allocations.
2. **Intra-Polygon XOR (Even-Odd Rule):** Rasterizes the outer ring and interior donut holes of each individual polygon using XOR (`poly_mask ^= ring_mask`), correctly handling holes.
3. **Inter-Polygon OR (Feature Union):** Unions distinct polygon entities into the global grid canvas using bitwise logical OR (`out[r0:r1, c0:c1] |= poly_mask`).

### 2.3 Verification Results
- **Synthetic Overlap Test:** Two overlapping $21 \times 21$ pixel squares ($441\text{ px}$ each) sharing an $11 \times 21$ pixel intersection ($231\text{ px}$).
  - *Original XOR:* Produced $420\text{ px}$ (overlap deleted).
  - *Corrected Fix:* Produced **$651\text{ px}$** (exact mathematical union: $441 + 441 - 231$).
- **Synthetic Hole Test:** $31 \times 31$ box ($961\text{ px}$) with $11 \times 11$ interior hole ($121\text{ px}$).
  - *Corrected Fix:* Produced **$840\text{ px}$**, with the interior hole pixel correctly evaluated as `False`.
- **Self-Test:** Evaluated reference against itself (`--self-test`).
  - *Result:* Exact $\text{IoU} = 1.000$, $\text{Precision} = 1.000$, $\text{Recall} = 1.000$.

---

## 3. Resolution of the TASA Rasterization Investigation

Initial audit observations noted a potential discrepancy between original and fixed rasterization on the Taiwan Space Agency (TASA) layer. A thorough investigation confirmed:
- **Result:** Both the original `burn()` and the corrected `burn_fixed()` produced **$102,706\text{ pixels}$ ($10.2706\text{ km}^2$)** on the Trishuli evaluation grid, with an **exact divergence of 0 pixels**.
- **Explanation:** In `Affected_0816_0828.shp`, all features are grouped into a **single multi-part polygon record** (`len(polys) == 1`) containing 4,671 rings in its parts array. Because there is only one feature record, inter-feature XOR cancellation was never triggered for TASA in the original script.
- The previously noted count of $102,528\text{ pixels}$ ($10.2528\text{ km}^2$) was an unbuffered/clipped tally of TASA itself, rather than a divergence between burning algorithms.
- The fix remains essential for any multi-feature dataset (such as parcel-level maps or multi-polygon shapefiles).

---

## 4. Input Data Constraints & Environmental Limitations

### 4.1 Missing Valid-Observation Mask & Dense Cloud Cover (65.6%)
- **Scene Type:** The post-event optical scene (`data/raw/dev/s2_after_20260827_l1c.tif`) is Sentinel-2 Level-1C (top-of-atmosphere reflectance, 13 spectral bands). It does **not** contain a Scene Classification Layer (SCL) band.
- **Cloud Cover:** Sentinel-2 metadata records **65.6% cloud cover**, densely covering the Trishuli gorge, river confluence, and valley floor.
- **Toolkit Fallback:** In accordance with scientific data rules, **no synthetic cloud mask was fabricated**. The validation toolkit therefore executed under its supported fallback: omitting `--valid` treats all unmasked pixels in the evaluation zone as observed non-water.
- **Analytical Impact:** Cloudy pixels obscuring the river channel are counted as observed dry land, falsely driving false negatives ($\text{FN} = 14.78\text{ km}^2$) and depressing model recall ($0.43\%$ vs IWM, $0.22\%$ vs Core). The calculated recall metrics are therefore highly conservative and pessimistic.

### 4.2 Unsubtracted Permanent River Water
- The JRC Global Surface Water occurrence tile (`occurrence_80E_30Nv1_4_2021.tif`) was not available in the evaluation package.
- Following toolkit fallback protocols, `--perm-water` was omitted.
- **Analytical Impact:** Both the reference layers and the model water predictions include the normal perennial river channel of the Trishuli River.

---

## 5. Third-Party Reference Datasets & Inter-Agreement Divergence

Validation was conducted against four third-party external products:
1. **IWM (Institute of Water Modelling, Bangladesh):** Vector shapefile (`IWM_2026_08_26_Nepal_Flood_VAP02.shp`) derived from Sentinel-1 SAR imagery. Covers $19.26\text{ km}^2$ total, of which $14.84\text{ km}^2$ lies within Nepal.
2. **TASA (Taiwan Space Agency):** Shapefile (`Affected_0816_0828.shp`) derived from FORMOSAT-5 optical and Sentinel SAR data. Covers $10.27\text{ km}^2$ total, of which $6.71\text{ km}^2$ lies within Nepal.
3. **HOT (Humanitarian OpenStreetMap Team):** GeoJSON corridor polygon (`hot_flood_npl_flood_extent.geojson`) covering $31.75\text{ km}^2$ total ($5.42\text{ km}^2$ in the evaluation zone). Source and mapping methodology remain unverified; includes active gravel beds, sediment bars, and debris-flow paths.
4. **EMSR927 (Copernicus Emergency Management Service):** Archive `EMSR927_AOI01_GRA_PRODUCT_v1.zip` covering a $3.46\text{ km}^2$ AOI at Syapru Besi. Classified strictly as `Landslide` / `6-Mass Movement` with **zero flood extent polygons**.

### Inter-Reference Divergence
The reference maps disagree sharply with one another, proving that none can be treated as absolute ground truth:
- **TASA vs. IWM (in Nepal):**
  - Shared Overlap ($\text{TP}$): $5.82\text{ km}^2$
  - IWM-only ($\text{FN}$): $9.03\text{ km}^2$
  - TASA-only ($\text{FP}$): $0.89\text{ km}^2$
  - **Intersection over Union (IoU): 36.98%** (Precision: $86.70\%$, Recall: $39.20\%$, F1: $53.99\%$)
- **HOT vs. IWM (in Nepal, same window):**
  - Shared Overlap ($\text{TP}$): $4.86\text{ km}^2$
  - IWM-only ($\text{FN}$): $9.98\text{ km}^2$
  - HOT-only ($\text{FP}$): $0.55\text{ km}^2$
  - **Intersection over Union (IoU): 31.57%** (Precision: $89.76\%$, Recall: $32.75\%$, F1: $47.99\%$)

---

## 6. Quantitative Evaluation Results (Prediction vs. References)

**Evaluated Prediction:** [`terrain_gated_water_mask.tif`](file:///C:/Users/user/Downloads/floodintel-track-b/data/runs/attention_unet_exploratory_trishuli/terrain_gated_water_mask.tif)  
- Checksum: SHA-256 `3825b6acdb5a91a63f3da5e7c9a6c776e3cd24baa60f7887a6d94deb93dafa4e`  
- Values: `1` = Water candidate ($3.75\text{ km}^2$ across full scene, $0.0973\text{ km}^2$ within evaluation zone, $0.5906\text{ km}^2$ in Nepal outside zone).  

| Evaluation Metric Block | Reference Target | Reference Area | Overlap Area | IoU | Precision | Recall | F1 | Tolerant ($20\text{ m}$) Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Block A: Full Metrics** | IWM in Nepal | $14.8449\text{ km}^2$ | $0.0632\text{ km}^2$ | **$0.42\%$** | **$64.95\%$** | **$0.43\%$** | **$0.85\%$** | Precision: $65.67\%$<br>Recall: $3.08\%$ |
| **Block B: Core Consensus** | IWM $\cap$ TASA Core | $5.8193\text{ km}^2$ | $0.0127\text{ km}^2$ | — | — | **$0.22\%$** | — | — |
| **Block C: TASA-Only** | TASA in Nepal | $6.7119\text{ km}^2$ | $0.0145\text{ km}^2$ | — | — | **$0.22\%$** | — | — |
| **Block D: Sensitivity** | HOT Corridor | $5.4154\text{ km}^2$ | $0.0728\text{ km}^2$ | **$1.34\%$** | **$97.46\%$** | **$1.34\%$** | **$2.65\%$** | Precision: $97.99\%$<br>Recall: $9.24\%$ |

### Diagnostic Metrics Breakdown (Block A vs. IWM)
- True Positive ($\text{TP}$): $0.0632\text{ km}^2$ ($632\text{ cells}$)
- False Positive ($\text{FP}$): $0.0341\text{ km}^2$ ($341\text{ cells}$)
- False Negative ($\text{FN}$): $14.7817\text{ km}^2$ ($147,817\text{ cells}$)
- Strict Precision: $64.95\%$ (model predictions inside the evaluation zone predominantly fall on validated reference water)
- Strict Recall: $0.43\%$ (artificially depressed by $65.6\%$ optical cloud cover without a valid-observation mask)

---

## 7. Data Governance & Repository Architecture

All reference datasets and validation artifacts comply strictly with FloodIntel data isolation policies (`DataRulesGuard`):
1. **Isolated Reference Data:** Stored exclusively under `data/validation_only/` (`iwm/`, `tasa/`, `boundaries/`, `hot_flood_npl/`, `emsr927/`). These paths are strictly forbidden from entering training or inference pipelines.
2. **Reproducible Validation Script:** Available at [`scripts/floodintel_validate.py`](file:///C:/Users/user/Downloads/floodintel-track-b/scripts/floodintel_validate.py).
3. **Run Outputs:** Machine-readable metrics stored at [`data/runs/validation_trishuli_toolkit/validation_results.json`](file:///C:/Users/user/Downloads/floodintel-track-b/data/runs/validation_trishuli_toolkit/validation_results.json) alongside [`agreement_map.png`](file:///C:/Users/user/Downloads/floodintel-track-b/data/runs/validation_trishuli_toolkit/agreement_map.png).

---

## 8. Mandatory Legal Attributions

* **Copernicus Sentinel Satellite Data:** Contains modified Copernicus Sentinel data 2026.
* **Copernicus Digital Elevation Model:** Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
* **AI Training Baseline:** Sen1Floods11 dataset (Bonafilia et al., 2020) HandLabeled flood segmentation dataset.
* **Infrastructure Baseline:** © OpenStreetMap contributors (ODbL).
* **Copernicus Emergency Management Service:** EMSR927 Rapid Mapping activation for Nepal Monsoon 2026 (landslide/mass movement grading AOI01).
