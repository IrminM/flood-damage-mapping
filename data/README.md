# data/ (git-ignored except this file)

Put local data here. Nothing in this folder is committed.

```
data/
  raw/<aoi>/        s1_before_YYYYMMDD.tif + .meta.json, s1_after_YYYYMMDD.tif + .meta.json, s1_candidates.json
  dem/<aoi>/        Copernicus DEM GLO-30 GeoTIFF(s)
  osm/<aoi>/        buildings/roads/bridges/places/health .geojson + snapshot_meta.json
  training/         small Kuro Siwo subset (prototype, ~100 MB)
  models/           best_flood_attention_unet.keras (152 MB trained checkpoint), baseline_unet.pth
  runs/<id>/        analysis outputs (rasters, GeoJSON, provenance.json, report)
```

Validation-only material (e.g. Copernicus EMS EMSR927 maps) must NOT be placed in any
input folder. Keep it under `data/validation_only/`; the pipeline's loaders reject
EMS/UNOSAT/damage-map names as inputs.
