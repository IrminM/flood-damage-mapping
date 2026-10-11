# FloodIntel: Satellite-Based Flood Impact & Connectivity Intelligence

**Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**

FloodIntel is an end-to-end, scientifically defensible geospatial intelligence platform. Given an area of interest and a flood event date, it synthesizes multimodal satellite observations (Sentinel-1 SAR and Sentinel-2 multispectral), Copernicus GLO-30 DEM topography, and pre-event OpenStreetMap infrastructure to answer four critical emergency response questions:

1. **Where did flood water and surface disruption occur?** Multi-sensor corroboration combining physical SAR backscatter change detection with an Attention U-Net deep learning water candidate segmentation model.
2. **What infrastructure was exposed?** Pre-event OpenStreetMap buildings, road segments, and bridge spans intersecting the validated hazard envelope.
3. **Which communities are cut off?** Road network reachability analysis modeling graph disruption and identifying isolated settlements that lose connectivity to nearest towns or regional hospitals.
4. **Where will flood waters drain downstream?** Topographical downstream flood path tracing modeled via D8 flow direction over Copernicus GLO-30 DEM.

---

## System Architecture & Pipeline Stages

The platform is organized into six strictly modular, reproducible stages:

| Stage | Domain | Description | Key Outputs |
|---|---|---|---|
| **Stage 1** | **Ingestion & Data Contracts** | Sentinel-1/2 metadata parsing, Copernicus DEM ingestion, pre-event OSM snapshot extraction (ohsome API), and strict data-rules enforcement. | Registered rasters, GeoJSON vectors, provenance sidecars. |
| **Stage 2** | **SAR Change Detection** | Radiometric calibration, speckle filtering, log-ratio change detection ($\Delta \text{dB}$), and physical terrain gating (slope exclusion, shadow/layover masks, HAND). | `change_classes.tif`, `log_ratio_db.tif`, `slope_deg.tif`, `hand_m.tif`. |
| **Stage 3** | **Multimodal AI & Fusion** | 15-channel Attention U-Net (`best_flood_attention_unet.keras`) inference on Sentinel-1 + Sentinel-2, physical terrain gating, and fusion with SAR temporal change. | `probability_exploratory.tif`, `raw_binary_water_mask.tif`, `terrain_gated_water_mask.tif`, `fused_evidence.tif`. |
| **Stage 4** | **Infrastructure Exposure** | Spatial overlay of pre-event OSM road networks, bridge spans, and building polygons with multi-sensor hazard corridor. | `affected_roads.geojson`, `affected_bridges.geojson`, `affected_buildings.geojson`, `infrastructure_summary.json`. |
| **Stage 5** | **Network Connectivity** | NetworkX road graph construction, hazard-based edge disruption, and Dijkstra reachability to nearest service towns and hospitals. | `settlement_connectivity.geojson`, `cut_off_settlements.geojson`, `connectivity_summary.json`. |
| **Stage 6** | **Downstream Flow Path** | D8 flow direction calculation over Copernicus GLO-30 DEM, hydrological channel tracing, and downstream settlement exposure order. | `downstream_path.geojson`, `downstream_summary.json`. |

---

## Disaster Case Studies & Multi-Event Switching

FloodIntel enforces **strict geospatial data isolation**: cached runs from one geographical basin are never displayed under another area name.

1. **Trishuli River Basin & Upper Gorge (Nepal, 2026-08-26) — `trishuli_basin` [Verified Baseline]**
   * High-relief Himalayan corridor affected by catastrophic monsoon flooding.
   * Fully processed real-data run: 35,117 baseline buildings, 5,180.5 km roads, 458 bridges.
   * Impacts: $0.46\text{ km}^2$ dual AI+SAR water, $12.65\text{ km}$ exposed roads, 29 exposed bridges, 49 settlements cut off.
   * AI Validation Status: `not_validated_on_Trishuli` (exploratory regime; uncalibrated on steep topography).
2. **Emilia-Romagna River Basins (Italy, 2023-05-17) — `emilia_romagna` [Pending Ingestion]**
   * Severe May 2023 riverine flooding across agricultural plains and Apennine foothills (EMSR659).
   * Status: `data_not_prepared`. Dashboard renders exact data requirements checklist (Sentinel-1 Track 44 pair, GLO-30 DEM tiles, 2023-05-01 OSM snapshot). No cached Trishuli rasters are leaked.
3. **Valencia DANA Flash Flooding (Spain, 2024-10-29) — `valencia_dana` [Pending Ingestion]**
   * October 2024 ravine and flash flooding across Rambla del Poyo and Turia basin (EMSR773).
   * Status: `data_not_prepared`. Dashboard displays missing data checklist (Sentinel-1 Track 8 pair, DEM tile, pre-event OSM).

---

## Independent Validation Benchmarks

* **Humanitarian OpenStreetMap Team (HOT) Floodway Benchmark**:
  * Evaluated against independently mapped active floodway corridor ($8.91\text{ km}^2$ in scene, observed 2026-08-27).
  * Dual-evidence corroboration achieves **95.1% precision** against observed active flood deposits.
  * Physical terrain gating eliminates **99.6% of steep hillside false alarms** caused by mountain shadow and layover artifacts.
* **Copernicus EMS (EMSR927 AOI01) Post-Hoc Infrastructure Comparison**:
  * Evaluated against EMSR927 grading package transportation layers.
  * Strictly kept for post-hoc comparison: EMSR927 observed events are mapped as landslide/mass-movement and are **not** used as flood-water training truth.
  * Preserves `validation_status: "not_validated_on_Trishuli"`.

---

## Quickstart & Installation

> **Windows Users:** See the dedicated [STARTUP_GUIDE_WINDOWS.md](STARTUP_GUIDE_WINDOWS.md) for 1-click startup (`run_dashboard.bat`) and troubleshooting common port/extraction issues.

### 1. Prerequisites
* Python 3.10 or higher (tested on Python 3.11 and 3.14 on Windows and Linux).
* Modern web browser (Chrome, Edge, Firefox).

### 2. Virtual Environment Setup

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Core scientific runtime + FastAPI backend (sufficient to run the backend and dashboard)
pip install -r requirements.txt

# Development and testing tools (pytest, httpx)
pip install -r requirements-dev.txt

# Optional: Multimodal AI segmentation & Attention U-Net training (PyTorch, TensorFlow, Keras)
pip install -r requirements-ml.txt
```

---

## Running the Platform

### Option A: 1-Click Startup (Windows)
Double-click `run_dashboard.bat` in the project root. It activates `.venv`, starts the backend on port 8000, and opens your default browser automatically.

### Option B: Single-Process Unified Startup (Command Line)

Start the FastAPI server (it serves both the REST API and the frontend dashboard):

```bash
python scripts/run_server.py
```

* **Interactive Command Console:** [http://127.0.0.1:8000/floodintel/](http://127.0.0.1:8000/floodintel/)
* **Interactive API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Backend Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

> **Important Note on `127.0.0.1`:** `127.0.0.1` refers to your local computer (localhost). Keep the terminal window running `python scripts/run_server.py` open while interacting with the dashboard in your browser. If the backend terminal is closed, the dashboard will display `"Unable to connect to backend service"`.

### Option C: Two-Process Startup (Development)

For frontend development or isolated service debugging:

**Terminal 1 (Backend API):**
```bash
python -m uvicorn floodmap.api.app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend Console):**
```bash
python -m http.server 3000 --directory frontend
```
Then visit [http://localhost:3000](http://localhost:3000) in your browser.

---

## Running Automated Tests

Run the full automated test suite (378 regression and unit tests):

```bash
# Full test suite
pytest

# API endpoints and layer retrieval tests
pytest tests/unit/test_api.py -v

# Disaster case studies & data isolation tests
pytest tests/unit/test_case_studies.py -v
```

---

## CLI Execution of Pipeline Stages

Each stage can be executed independently via command-line scripts:

```bash
# Stage 2: Sentinel-1 SAR change detection & terrain gating
python scripts/run_change_detection.py --config config/default.yaml

# Stage 3: Attention U-Net inference & multi-evidence fusion
python scripts/run_stage3.py --config config/default.yaml

# Stage 4: OpenStreetMap infrastructure exposure assessment
python scripts/run_stage4.py --config config/default.yaml

# Stage 5: Road network connectivity & cut-off community analysis
python scripts/run_stage5.py --config config/default.yaml

# Stage 6: Copernicus DEM downstream drainage path tracing
python scripts/run_stage6.py --config config/default.yaml

# Independent validation benchmarks
python scripts/compare_hot_validation.py
python scripts/compare_emsr927_infrastructure.py
```

---

## Project Directory Structure

```
floodintel-track-b/
├── config/
│   ├── default.yaml                             # Master thresholds and pipeline parameters
│   ├── attention_unet_s2_provenance.yaml        # AI sensor calibration parameters
│   └── aoi_examples/                            # Example AOI GeoJSON boundaries
├── data/
│   ├── models/                                  # Trained Attention U-Net weights (.keras)
│   ├── raw/dev/                                 # Sentinel-1 & Sentinel-2 input scenes
│   ├── dem/dev/                                 # Copernicus GLO-30 DEM raster and tiles
│   ├── osm/dev/                                 # Pre-event OpenStreetMap baseline vectors
│   ├── validation_only/                         # EMSR927 and HOT reference packages
│   └── runs/                                    # Verified assessment outputs and reports
├── dev_mocks/                                   # Deterministic synthetic fixtures for pytest
├── docs/                                        # Scientific architecture, data rules, attributions
├── floodmap/                                    # Scientific Python package and FastAPI backend
│   ├── api/                                     # FastAPI application, routers, and endpoints
│   ├── core/                                    # Contracts, GeoRaster, rules, case registry
│   ├── detection/                               # SAR change detection algorithms
│   ├── ai/                                      # Multimodal Attention U-Net adapters & inference
│   ├── gis/                                     # Vectorization, overlays, spatial calculations
│   ├── network/                                 # NetworkX road graph & reachability analysis
│   ├── satellite/                               # S1/S2 preprocessing & terrain gating
│   └── dem/                                     # Topographical D8 downstream tracing
├── frontend/                                    # Standalone Command Console web app
│   ├── index.html                               # Dashboard markup & Leaflet viewports
│   ├── css/                                     # Design tokens, layout, components, print
│   ├── js/                                      # Modular API, UI, map, and findings logic
│   └── vendor/leaflet/                          # Offline Leaflet library (zero CDN dependency)
├── scripts/                                     # CLI runners for all pipeline stages & benchmarks
├── tests/unit/                                  # 47 unit test suites (378 passing tests)
├── pyproject.toml                               # Build system & pytest configuration
├── requirements.txt                             # Production backend dependencies
├── requirements-dev.txt                         # Testing dependencies (pytest, httpx)
├── requirements-ml.txt                          # Machine learning dependencies (TensorFlow, PyTorch)
└── README.md                                    # This documentation
```

---

## Downstream Drainage-Flow Bonus Feature

FloodIntel includes a deterministic hydrological downstream flow path tracer derived from terrain topography:
* **Algorithm & Elevation Model:** Computes a downhill D8 priority-flood drainage corridor on Copernicus WorldDEM GLO-30 (30 m resolution), tracing steepest descent from the upper Rasuwa gorge (`[85.25, 28.15]`).
* **Path Metrics:** Extends **50.01 km** downstream with a **2,551 m vertical drop** (starting elevation 3,117 m down to 566 m; 1,398 DEM grid steps).
* **Downstream Settlement Exposure:** Identifies and sequences **20 pre-event OSM settlements** situated within a 1,000 m proximity buffer along the drainage centerline (e.g., Syabrubesi → Dhunche → Ramche → Betrawati → Trishuli Bazar).
* **Hydrological Caveat:** This layer models a *topographic gravitational drainage pathway*. It is **not** an observed floodwater polygon or hydrodynamic wave velocity simulation; landslide damming, river avulsions, and debris clogging are not modeled.

---

## Live-Evaluation Readiness & Supported Scope

Track B operational capabilities are categorized into three verified tiers:
1. **Verified & Operational (Trishuli River Basin, 2026-08-26):**
   * Fully processed multimodal pipeline with Sentinel-1 SAR change detection, Copernicus DEM-30 terrain gating, and pre-event OSM road graph routing (7,188 road segments, 458 bridges, 505 settlements).
   * Third-party reference agreement evaluation vs IWM (Bangladesh) and TASA (Taiwan) via `scripts/floodintel_validate.py`. Formal status: `not_validated_on_Trishuli`.
2. **Supported Ingestion Contracts (Emilia-Romagna 2023 & Valencia 2024):**
   * Pre-configured case studies with verified geographic bounding boxes and metadata.
   * Frontend and backend safely display `"Data Not Prepared: Pending Ingestion"` with zero cross-AOI leakage.
3. **Arbitrary Areas & Dates (Live External Downlink):**
   * Arbitrary global coordinates or dates outside the August 2026 monsoon window are rejected with clear, explanatory HTTP 400 responses.
   * On-the-fly live querying of Copernicus Data Space Ecosystem (CDSE) / openEO and global ohsome PBF parsing requires cloud API keys and external network connectivity not packaged inside an offline ZIP.

---

## Scientific Limitations & Data Rules

1. **Exploratory AI Regime:** The Attention U-Net was trained on the Sen1Floods11 lowland plains dataset. It is uncalibrated on steep Himalayan topography. Its predictions are explicitly designated as exploratory candidates (`validation_status: "not_validated_on_Trishuli"`).
2. **Hazard Corridor vs. Confirmed Damage:** Intersecting OpenStreetMap features represent exposure to satellite-detected water and surface change envelopes. Proximity indicates potential disruption; physical structural collapse requires on-site structural engineering verification.
3. **SAR Non-Water Surface Change:** Radar backscatter temporal drops in steep valley bottoms reflect surface roughness, erosion, or soil moisture shifts. They are radar backscatter anomalies, not field-verified sediment deposits.
4. **Strict Pre-Event Data Boundary:** All infrastructure baselines are derived exclusively from pre-event OSM snapshots. No post-event OSM edits, Copernicus EMS damage layers, or UNOSAT maps are used as pipeline inputs.

---

## Data Sources & Mandatory Attributions

* **Copernicus Sentinel Satellite Data:** Contains modified Copernicus Sentinel data 2026.
* **Copernicus Digital Elevation Model:** Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
* **AI Training Baseline:** Sen1Floods11 dataset (Bonafilia et al., 2020) HandLabeled flood segmentation dataset. Evaluated as an exploratory baseline; distinct from Trishuli local validation.
* **Infrastructure Baseline:** © OpenStreetMap contributors (ODbL), extracted via HeiGIT ohsome API using historical snapshot prior to event date.
* **Meteorological Context:** Open-Meteo.com weather archive & forecast API (CC BY 4.0).
* **Validation References:** Copernicus Emergency Management Service Rapid Mapping (EMSR927 Rapid Mapping activation for Nepal Monsoon 2026, landslide/mass movement grading AOI01); Humanitarian OpenStreetMap Team (HOT) Nepal Flood Activation; Institute of Water Modelling (IWM Bangladesh); Taiwan Space Agency (TASA). Reference datasets are strictly isolated under `data/validation_only/` and never used for model training or inference.
