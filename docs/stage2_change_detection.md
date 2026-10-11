# Stage 2: Sentinel-1 preprocessing, terrain masks and baseline change detection

This is **traditional remote sensing (path A)**. It contains no AI. AI segmentation (path B) comes in stage 3,
behind its own interface, and the two are fused later with per-pixel evidence provenance.

## Pipeline

```
before.tif + .meta.json ─┐
after.tif  + .meta.json ─┼─ load + data-rules guard ─ pair check (same relative orbit & direction)
DEM (optional)          ─┘
        │
AOI ─ analysis grid (auto UTM, 10 m, snapped) + AOI mask
        │
preprocess_s1:  units → linear · resample (bilinear, linear power) · Lee 5×5 · → dB · valid range
        │        shared valid mask = valid(before) ∧ valid(after) ∧ inside AOI
terrain (30 m): slope/aspect · shadow/layover (local approx.) · priority-flood + D8 → HAND
        │        resampled to the analysis grid
s1_logratio_v1: log-ratio VV/VH · dark-water threshold (bounded Otsu) · classes · terrain gates · clean-up
        │
outputs: change_classes.tif · log_ratio_db.tif · exclusion_flags.tif · terrain/* · summary.json
         · quicklooks/*.png · provenance.json
```

## Classes (`change/change_classes.tif`, uint8)

| Code | Name | Rule |
|---|---|---|
| 0 | no detected change | observable, none of the rules below |
| 1 | new open water | after VV ≤ T, before VV > T, VV drop ≥ 3 dB, passes slope ≤ 15° and HAND ≤ 25 m |
| 2 | water in both dates | before and after VV ≤ T, same terrain gate |
| 3 | non-water backscatter change | \|ΔVV\| or \|ΔVH\| ≥ change threshold (per polarisation, see below), not water after, slope ≤ 35° |
| 255 | not observable | outside AOI, no data in either date, radar shadow/layover |

T is the dark-water threshold. Otsu runs on after-event VV for observable, water-plausible pixels, and
the Otsu value is **accepted only if it is a credible land/open-water split**:
(a) it lies inside [−24, −14] dB, and (b) the two halves are clearly bimodal (Ashman's D ≥ 3.5; a
unimodal histogram split by Otsu already gives D ≈ 2.65, land plus ≥10 % open water gives D ≈ 6).
Otherwise the conservative fixed **−18 dB** is used (`fixed_fallback`). Clamping to the nearest bound
exists only as an explicit `otsu_out_of_bounds: clamp` policy. The value, raw Otsu, Ashman D and method
are written to `summary.json` and `provenance.json`.

*Why this changed (first real run, Trishuli corridor, 16→28 Aug 2026):* Otsu landed at −7.61 dB, a split
between bright and dark **land** (slopes facing towards/away from the radar; `sigma0-ellipsoid` is not
terrain-flattened), not between land and water. The original code clamped this to −14 dB, the most
permissive water threshold. Global Otsu also cannot isolate water covering only a few percent of a scene,
such as a narrow, turbulent Himalayan river, which may not even be dark. **Split-based (tile-wise)
thresholding** is the standard upgrade and can replace `water_threshold` behind the same interface.

### Change threshold (noise-adaptive) and terrain context

**Threshold.** For each polarisation the class-3 threshold is `max(change_abs_db, k_sigma × σ)`,
where σ is the robust (1.4826·MAD) standard deviation of that pair's log-ratio over observable non-water
pixels (`change_threshold.mode: noise_adaptive`, `k_sigma: 3`, floor `change_abs_db: 3 dB`).
`mode: fixed` restores a plain 3 dB. *Why:* on the real Trishuli pair σ ≈ 1.45 dB, so a fixed 3 dB sat at
about 2σ, and roughly 8 % of non-water pixels exceeded it from noise alone. On clean scenes 3σ falls below
the floor and results are identical to the fixed mode (tested). MAD is biased slightly upward by one-sided
real change, which errs conservative. The values used are in `summary.json → thresholds.change`.

**Terrain context** (`change/change_context.tif`, `quicklooks/change_context.png`,
`summary.json → class3_context_km2`). Each class-3 pixel is labelled **valley bottom**
(HAND ≤ 25 m and slope ≤ 15°, `change_context.*` in config), **hillside**, or **unknown** (no DEM).
These are labels only; no class changes. Valley-bottom change is where floods and debris flows deposit,
so it is the *candidate event-related* part. It stays unvalidated until checked against a pre-event null
pair or an independent reference. Codes: 0 no class-3 · 1 valley bottom · 2 hillside · 3 unknown · 255 nodata.

### Non-water change diagnostics (`summary.json → detection_stats.diagnostics`)

These are evidence only; they do not alter any class:
* **noise:** robust σ (1.4826·MAD) of the VV/VH log-ratio over observable non-water pixels, the observed
  share with |log-ratio| ≥ threshold, and a Gaussian lower bound. The Gaussian figure underestimates
  speckle tails. In a synthetic no-change pair it predicts ~0 % while ~0.35 % exceed 3 dB before clean-up,
  and **0 %** survive clean-up.
* **sign:** share of class 3 that brightened vs darkened.
* **terrain enrichment:** class-3 share per HAND and slope band vs that band's share of the area.
  Valley-bottom processes (flood, debris deposition) should be enriched at low HAND; noise and vegetation
  change should not.

### How class 3 must be validated before it is interpreted
1. **Null (pre-event) pair:** run the identical pipeline on two pre-event acquisitions from the same
   track (e.g. 4→16 Aug). The class-3 area there is the method's real false-alarm level for this terrain
   and season. Post-event class 3 is meaningful only to the extent it clearly exceeds that.
2. **Spatial plausibility:** enrichment at low HAND along the river corridor, visible in the quicklooks.
3. **Independent check (validation only, never input):** at the case-study stage, compare with the
   EMSR927 reference maps. They are not used for tuning.

`exclusion_flags.tif` records *why* a pixel was not classified or was rejected. Bit flags: 1 outside AOI,
2 no data, 4 radar shadow, 8 radar layover, 16 water rejected by the terrain gate, 32 change rejected by
the terrain gate, 64 removed as a small blob.

## How to interpret the classes (keep this wording in reports)

* **Class 1 is open water only.** Flooded vegetation and flooded buildings get brighter (double bounce)
  and are not captured as water.
* **Class 3 is not debris.** It is any strong non-water backscatter change: debris/mud deposits, damaged
  structures and landslides, but also vegetation, crop and moisture change, plus residual speckle. It is
  evidence for later fusion and review, not a debris map.

## Assumptions and approximations

1. Input is already calibrated backscatter on a map grid: the CDSE openEO `sar_backscatter` output with
   coefficient `sigma0-ellipsoid`, the only coefficient that backend accepts. It is **not radiometrically
   terrain-flattened**, so slopes facing the radar look brighter and slopes facing away look darker. The
   DEM-based shadow/layover and slope/HAND gates below exist partly for this reason, and the coefficient is
   recorded in each scene sidecar and in provenance. Raw GRD processing is not done here.
2. Before and after scenes must share relative orbit and pass direction. This is enforced, and
   configurable but on by default, as the brief requires.
3. Resampling is done in linear power. The Lee filter runs after alignment, on the 10 m grid.
4. Shadow/layover use each pixel's local slope with a nominal incidence angle (39°, or
   `extra.incidence_angle_deg` from the sidecar) and a nominal heading (asc 348°, desc 192°, right-looking).
   There is no ray tracing, so shadows cast by distant ridges are missed.
5. HAND only knows drainage inside the DEM extent (AOI + `terrain.margin_m`). Near the DEM edge, HAND can
   be unknown (NaN). An unknown HAND does not block the water gate; the slope gate still applies.
6. All thresholds are untuned defaults in `config/default.yaml`. **They will not be tuned on EMSR927**,
   which may only be used for checking. Tuning will use permitted training labels (e.g. Kuro Siwo).

## Verification status

| Part | Synthetic (SYNTHETIC_TEST) | Real Sentinel-1 |
|---|---|---|
| grid, AOI mask, reprojection | tested | **not yet** |
| dB/linear handling, Lee filter, pair preprocessing | tested | **not yet** |
| slope/aspect, shadow/layover, HAND | tested (analytic planes, synthetic valley) | **not yet** (needs real DEM) |
| change detection classes, gates, clean-up | tested against known truth | **not yet** |
| pipeline, CLI, summary, provenance | tested end to end | **not yet** |
| `fetch_dem_glo30.py` download | fake tile server: mosaic across tile seam, pixel alignment, retries, truncation, missing tiles (nodata, not 0 m), cache validation | **not yet run** |
| lon/lat (EPSG:4326) DEM → UTM terrain analysis | tested: known 3° slope recovered after reprojection | **not yet** |
| real Sentinel-1 input | – | runs v1/v2 completed (16→28 Aug, orbit 85 asc). v1 exposed the Otsu clamping flaw (fixed). v2 diagnostics: class 3 is 5× enriched at HAND 0–10 m, but 66 % lies on hillsides with σ ≈ 1.45 dB → noise-adaptive threshold and terrain context added; v3 rerun pending |
