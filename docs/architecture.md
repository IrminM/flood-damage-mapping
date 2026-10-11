# Architecture

```
Frontend (Leaflet, no science)
   │ HTTP/JSON
FastAPI backend (routes, jobs, result store)
   │ AnalysisRequest
Orchestrator (floodmap.pipeline) — runs steps, saves intermediates, provenance, data-rules guard
   ├── data_sources   S1/S2 scenes, DEM, pre-event OSM (local files now; CDSE download later)
   ├── satellite      pairing rules, clip/align/normalise, terrain masks
   ├── detection      A) change detection   B) AI segmentation (via ai/ registry)   → fusion
   ├── gis            vectorise flood layer; affected buildings / roads / bridges
   ├── network        road graph, disrupted edges, settlement → town/hospital connectivity
   └── dem (optional) upstream point → approximate downstream path → settlements
   │
AnalysisResult (stats + GeoJSON + rasters + provenance) → dashboard + situation report
```

## Principles

* **The scientific core (`floodmap/`) has no web code.** The backend and frontend are thin
  layers, and the same pipeline runs from the CLI and from tests.
* **Contracts, not internals.** Modules exchange only the objects in
  `floodmap/core/contracts.py` and the `GeoRaster` type. Replacing an implementation
  (better preprocessing, the teammate's U-Net, a better connectivity algorithm) must not
  change those shapes.
* **One canonical class map** after segmentation: 0 background, 1 flood water,
  2 permanent water, 3 debris/surface change, 255 nodata. Model adapters map onto it.
* **Per-pixel evidence provenance**: change detection (1), AI (2), both (3).
* **Every threshold lives in `config/default.yaml`.**
* **Data rules are enforced in code** (`floodmap/core/data_rules.py`). See `data_rules.md`.
* **Mocks are isolated** in `dev_mocks/`. A test fails if `floodmap/` imports them, and
  any synthetic input stamps the run `data_mode=SYNTHETIC_TEST`.

## Module status

| Stage | Module | Status |
|---|---|---|
| 1 | config, contracts, GeoRaster, geo utils, data rules, provenance | done |
| 1 | local loaders (S1/S2 scenes + sidecar, DEM, OSM), pairing rules | done |
| 1 | ohsome + CDSE catalogue clients, fetch scripts | done (network parts untested in sandbox) |
| 2 | preprocessing, terrain masks, change detection | built; synthetic-tested; real-data run pending |
| 3 | AI interface, baseline model, teammate adapter slot, fusion | planned |
| 4 | GIS impact, road network, cut-off | planned |
| 5 | orchestrator + CLI, FastAPI, dashboard, situation report | planned |
| 6 | DEM downstream path (optional) | planned |
