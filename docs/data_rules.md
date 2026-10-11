# Data rules (Track B), and how the code enforces them

| Rule (from the challenge brief) | Enforcement |
|---|---|
| Inputs: Sentinel-1, Sentinel-2, Copernicus DEM, pre-event OSM, listed training datasets | Every loader declares a `SourceType`; `DataRulesGuard.check_input` rejects anything not in `data_rules.allowed_source_types` |
| Copernicus EMS, UNOSAT or any published damage map must NOT be an input | Paths, product IDs and processing text are matched against `data_rules.prohibited_source_patterns` (EMS, EMSR, UNOSAT, damage assessment, grading, delineation...). A match raises `DataRuleViolation` and stops the run. |
| No OSM edits made after the event | OSM is fetched from ohsome *at a snapshot date*. `snapshot_meta.json` records it, and the loader refuses `snapshot_date >= event_date`. The fetch script refuses before calling the API. |
| Same orbit track for change detection | `satellite/pairing.py` rejects pairs with a different relative orbit or pass direction (configurable, on by default) |
| EMSR927 may be used only for checking | Keep it in `data/validation_only/`. A separate comparison script (later stage) reads it. Pipeline loaders never do. |

Every run writes `provenance.json` (inputs with SHA-256, snapshot dates, processing steps).
That file is the evidence that no prohibited inputs were used.
