# FloodIntel — Geospatial Command Console (Frontend)

Light-theme interactive disaster intelligence console for satellite-based flood impact, infrastructure exposure, and community connectivity assessment (Multimodal AI Hackathon 2026, Track B).

Built with plain HTML5, CSS3, and vanilla JavaScript (ES6+). Includes offline-bundled Leaflet GIS assets under `vendor/leaflet/` (zero Node.js or CDN build requirements).

---

## Quickstart

### 1. Served via Unified FastAPI Backend (Recommended)
When running the backend from the project root (`python scripts/run_server.py`), the console is mounted automatically at:

**[http://127.0.0.1:8000/floodintel/](http://127.0.0.1:8000/floodintel/)**

### 2. Standalone Static Server
To run the frontend on a dedicated port:

```bash
# From the project root:
python -m http.server 3000 --directory frontend

# Or from within this frontend directory:
python -m http.server 3000
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser. It connects automatically to the FastAPI backend running on port 8000.

---

## Directory Layout

```
frontend/
├── index.html                 # Main Geospatial Command Console workspace
├── css/
│   ├── base.css               # Design tokens, typography, CSS custom properties
│   ├── layout.css             # 2-column workspace grid, header, toolbar, responsive
│   ├── components.css         # KPI cards, floating GIS layer legend, badges, tables
│   └── print.css              # Formal disaster Situation Report print formatting
├── js/
│   ├── config.js              # Application settings, API endpoint discovery, layers
│   ├── utils.js               # Validation and DOM utility helpers
│   ├── main.js                # App entrypoint, state lifecycle, run dispatch
│   ├── api/
│   │   ├── assessmentApi.js   # Client for POST /api/assess and GET /api/cases
│   │   └── weatherApi.js      # Open-Meteo API meteorological client
│   └── ui/
│       ├── controls.js        # Tab navigation, case study selector, review modal
│       ├── map.js             # Leaflet GIS engine, GeoJSON styling, layer groups
│       ├── findings.js        # Multi-sensor KPI cards, settlement table, data-not-prepared state
│       ├── report.js          # Disaster Situation Report (SitRep) generator
│       └── weatherPanel.js    # Atmospheric weather chip rendering
└── vendor/
    └── leaflet/               # Offline Leaflet CSS, JS, and marker assets
```

---

## Disaster Case Studies

The console features a multi-event case study selector wired to backend data isolation:
* **Trishuli Basin, Nepal (2026-08-26)**: Verified full-scene assessment with dual AI+SAR water, exposed OSM roads/bridges, and 49 cut-off settlements.
* **Emilia-Romagna, Italy (2023-05-17)**: Pending ingestion; displays exact missing inputs checklist (Sentinel-1 Track 44 pair, GLO-30 DEM, pre-event OSM).
* **Valencia DANA, Spain (2024-10-29)**: Pending ingestion; displays missing inputs checklist.

---

## Attributions
* **Basemaps:** OpenStreetMap contributors (ODbL) & Esri World Imagery.
* **Satellite Observations:** Modified Copernicus Sentinel data (2026), ESA.
* **Weather Data:** Open-Meteo.com (CC BY 4.0).
