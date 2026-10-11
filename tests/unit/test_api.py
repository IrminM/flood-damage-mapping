"""Integration tests for FastAPI endpoints and end-to-end orchestration."""
from datetime import date
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from dev_mocks.synthetic_scene import synthetic_aoi, write_synthetic_pair
from floodmap.api.app import app
from floodmap.core.contracts import (
    AnalysisStatus,
    DataAvailabilityStatus,
    DataMode,
)

EVENT = date(2026, 8, 26)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synth_setup(tmp_path):
    bpath, apath, pair = write_synthetic_pair(tmp_path / "in")
    aoi = synthetic_aoi(pair, inset_m=30)
    dem_path = tmp_path / "in" / "SYNTHETIC_dem.tif"
    return bpath, apath, dem_path, aoi


def test_root_and_health_endpoints(client):
    """Verify root index, health check, and model inspection endpoints."""
    # 1. Root index
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Flood Damage Mapper API"
    assert "/docs" in data["documentation"]

    # 2. Health check
    h_resp = client.get("/api/v1/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "healthy"
    assert "Stage 1" in h_data["stages_supported"]
    assert "Stage 6" in h_data["stages_supported"]
    assert h_data["ai_engine"]["active_model"] == "s1_mini_unet_v1"
    assert "baseline" in h_data["ai_engine"]["role"].lower()
    assert h_data["data_rules"]["enforced"] is True

    # 3. Model A status
    m_resp = client.get("/api/v1/model-a/status")
    assert m_resp.status_code == 200
    m_data = m_resp.json()
    assert m_data["model_name"] == "model_a_custom_unet"
    assert m_data["operational_readiness"] == "DISABLED_PENDING_PROVENANCE"

    # 3b. Attention U-Net status
    att_resp = client.get("/api/v1/attention-unet/status")
    assert att_resp.status_code == 200
    att_data = att_resp.json()
    assert att_data["model_name"] == "Attention_UNet_S1_S2"
    assert att_data["loadable"] is True
    assert att_data["operational_readiness"] == "BLOCKED_PENDING_VALID_S2_INPUT"
    assert att_data["decision_threshold"] == 0.40
    assert att_data["checkpoint_exists"] is True

    att_alias_resp = client.get("/api/attention-unet/status")
    assert att_alias_resp.status_code == 200

    # 4. Data readiness
    d_resp = client.get("/api/v1/data-status")
    assert d_resp.status_code == 200
    d_data = d_resp.json()
    assert "sentinel_1" in d_data
    assert "dem" in d_data
    assert "osm" in d_data


def test_submit_analysis_without_osm_data_explicit_unavailability(client, synth_setup, tmp_path):
    """Verify that when OSM data is not provided, stages 4 and 5 are explicitly unavailable, NOT false zeros."""
    bpath, apath, dem_path, aoi = synth_setup

    payload = {
        "aoi": aoi.geometry,
        "event_date": "2026-08-26",
        "before_scene": str(bpath),
        "after_scene": str(apath),
        "dem": str(dem_path),
        "osm_dir": str(tmp_path / "nonexistent_osm"),
        "name": "Synthetic No-OSM Test",
        "sync": True,
    }

    resp = client.post("/api/v1/analyses", json=payload)
    assert resp.status_code == 200
    res = resp.json()

    assert res["status"] == "completed"
    assert res["data_mode"] == "SYNTHETIC_TEST"
    assert res["stages_executed"]["stage1_sar_ingestion"] == "COMPLETED"
    assert res["stages_executed"]["stage2_change_detection"] == "COMPLETED"
    assert res["stages_executed"]["stage3_ai_fusion"] == "COMPLETED"

    # Explicit data availability guarantees
    assert res["data_availability"]["osm_preevent"] == "UNAVAILABLE"
    assert "missing" in res["data_availability"]["osm_unavailable_reason"].lower() or "not" in res["data_availability"]["osm_unavailable_reason"].lower()

    # Stages 4 and 5 must be marked unavailable, NOT fabricated 0 results
    assert res["stages_executed"]["stage4_infrastructure"] == "UNAVAILABLE_NO_PREEVENT_OSM"
    assert res["stages_executed"]["stage5_connectivity"] == "UNAVAILABLE_NO_ROAD_NETWORK"
    assert res["stage4_summary"] is None
    assert res["stage5_summary"] is None

    # Check limitations notes
    limitations = " ".join(res["uncertainty_and_limitations"])
    assert "STAGE 4 NOTE" in limitations
    assert "STAGE 5 NOTE" in limitations

    # Verify retrieval via GET /api/v1/analyses/{id}
    aid = res["analysis_id"]
    get_resp = client.get(f"/api/v1/analyses/{aid}")
    assert get_resp.status_code == 200
    assert get_resp.json()["analysis_id"] == aid

    # Verify Situation Report
    sitrep_resp = client.get(f"/api/v1/analyses/{aid}/sitrep")
    assert sitrep_resp.status_code == 200
    sitrep = sitrep_resp.json()
    assert sitrep["infrastructure_findings"]["status"] == "UNAVAILABLE"
    assert sitrep["connectivity_findings"]["status"] == "UNAVAILABLE"

    # Verify layer retrieval
    assert "change_classes" in res["available_layers"]
    layer_resp = client.get(f"/api/v1/analyses/{aid}/layers/change_classes")
    assert layer_resp.status_code == 200
    assert len(layer_resp.content) > 0


def test_submit_analysis_with_osm_and_downstream(client, synth_setup, osm_dir_factory, tmp_path):
    """Verify full end-to-end pipeline execution with OSM and downstream tracing."""
    bpath, apath, dem_path, aoi = synth_setup
    osm_path = osm_dir_factory(tmp_path / "osm_data")

    payload = {
        "aoi": aoi.geometry,
        "event_date": "2026-08-26",
        "before_scene": str(bpath),
        "after_scene": str(apath),
        "dem": str(dem_path),
        "osm_dir": str(osm_path),
        "upstream_point": [float(aoi.geometry["coordinates"][0][0][0]), float(aoi.geometry["coordinates"][0][0][1])],
        "name": "Full Multi-Stage Integration Test",
        "sync": True,
    }

    resp = client.post("/api/v1/analyses", json=payload)
    assert resp.status_code == 200
    res = resp.json()

    assert res["status"] == "completed"
    assert res["stages_executed"]["stage1_sar_ingestion"] == "COMPLETED"
    assert res["stages_executed"]["stage2_change_detection"] == "COMPLETED"
    assert res["stages_executed"]["stage3_ai_fusion"] == "COMPLETED"
    assert res["stages_executed"]["stage4_infrastructure"] == "COMPLETED"
    assert res["stages_executed"]["stage5_connectivity"] == "COMPLETED"
    assert res["stages_executed"]["stage6_downstream"] == "COMPLETED"

    # Check summaries exist
    assert res["stage4_summary"] is not None
    assert res["stage5_summary"] is not None
    assert res["stage6_summary"] is not None

    aid = res["analysis_id"]
    # Check sitrep
    sitrep_resp = client.get(f"/api/v1/analyses/{aid}/sitrep")
    assert sitrep_resp.status_code == 200
    sitrep = sitrep_resp.json()
    assert sitrep["infrastructure_findings"]["status"] == "COMPLETED"
    assert sitrep["connectivity_findings"]["status"] == "COMPLETED"
    assert sitrep["downstream_findings"]["status"] == "COMPLETED"

    # Check layer retrieval for vector GeoJSON
    assert "infra_affected_buildings" in res["available_layers"]
    layer_resp = client.get(f"/api/v1/analyses/{aid}/layers/infra_affected_buildings")
    assert layer_resp.status_code == 200
    assert layer_resp.headers["content-type"] == "application/geo+json"


def test_api_error_cases(client):
    """Verify 404 responses for nonexistent resources."""
    resp = client.get("/api/v1/analyses/nonexistent_123")
    assert resp.status_code == 404

    resp2 = client.get("/api/v1/analyses/nonexistent_123/sitrep")
    assert resp2.status_code == 404

    resp3 = client.get("/api/v1/analyses/nonexistent_123/layers/fake_layer")
    assert resp3.status_code == 404


def test_unversioned_routes_and_aliases(client):
    """Verify /api/health, /api/analysis, /api/report, and /api/layers endpoint aliases."""
    # Health alias
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "healthy"

    # Data readiness alias
    d_resp = client.get("/api/data-status")
    assert d_resp.status_code == 200

    # Default/latest analysis
    a_resp = client.get("/api/analysis")
    assert a_resp.status_code == 200
    a_data = a_resp.json()
    assert "analysis_id" in a_data
    assert a_data["status"] == "completed"

    # Default/latest situation report
    r_resp = client.get("/api/report")
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert "executive_summary" in r_data
    assert "key_findings" in r_data

    # List layers
    l_resp = client.get("/api/layers")
    assert l_resp.status_code == 200
    layers = l_resp.json()
    assert isinstance(layers, dict)
    assert len(layers) > 0

    # Layer aliases on latest analysis
    for alias_name in ["cut_off_settlements", "affected_roads", "affected_buildings", "downstream_path", "flood_evidence"]:
        layer_file_resp = client.get(f"/api/layers/{alias_name}")
        assert layer_file_resp.status_code == 200, f"Layer {alias_name} should resolve"
        assert len(layer_file_resp.content) > 0


def test_real_data_smoke_run_retrieval(client):
    """Verify smoke_stage3_real analysis retrieval with real metrics."""
    aid = "smoke_stage3_real"

    # Full analysis result
    res_resp = client.get(f"/api/v1/analyses/{aid}")
    assert res_resp.status_code == 200
    res = res_resp.json()
    assert res["data_mode"] == "REAL"
    assert res["status"] == "completed"

    # Verify Stage 4 real stats
    assert res["stage4_summary"]["buildings"]["potentially_affected_count"] == 1101
    assert res["stage4_summary"]["bridges"]["potentially_affected_count"] == 41
    assert round(res["stage4_summary"]["roads"]["potentially_affected_length_km"], 2) == 18.96

    # Verify Stage 5 real stats
    assert res["stage5_summary"]["disconnected_count"] == 343
    assert res["stage5_summary"]["total_settlements"] == 505

    # Verify Stage 6 real stats
    assert round(res["stage6_summary"]["path_length_km"], 2) == 50.01
    assert res["stage6_summary"]["settlements_count"] == 20

    # Situation report
    sitrep_resp = client.get(f"/api/v1/analyses/{aid}/sitrep")
    assert sitrep_resp.status_code == 200
    sitrep = sitrep_resp.json()
    assert sitrep["infrastructure_findings"]["buildings_potentially_affected"] == 1101
    assert sitrep["connectivity_findings"]["settlements_cut_off"] == 343
    assert sitrep["downstream_findings"]["settlements_along_path_count"] == 20


def test_post_assess_success_trishuli_corridor(client):
    """Verify POST /api/assess and /api/v1/assess return valid contract for supported area & date."""
    payload = {
        "area": "Bhote Koshi–Trishuli corridor, Nepal",
        "date": "2026-08-26",
    }

    # Test /api/assess (frontend endpoint)
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Core metadata
    assert data["status"] == "completed"
    assert data["validation_status"] == "not_validated_on_Trishuli"
    assert data["data_mode"] == "REAL"
    assert data["area"] == "Bhote Koshi–Trishuli corridor, Nepal"
    assert data["date"] == "2026-08-26"

    # Compatibility cards
    assert data["floodAreaKm2"] == 0.4560
    assert data["debrisAreaKm2"] == 1.0856

    # Granular summary counts
    summary = data["summary"]
    assert summary["buildings_affected"] == 873
    assert summary["buildings_total"] == 35117
    assert summary["roads_affected_segments"] == 62
    assert summary["roads_total_segments"] == 7188
    assert summary["roads_affected_km"] == 12.65
    assert summary["bridges_affected"] == 29
    assert summary["bridges_total"] == 458
    assert summary["settlements_cut_off"] == 49
    assert summary["settlements_connected"] == 295
    assert summary["settlements_total"] == 505

    # Hazard metrics
    hazard = data["hazard_metrics"]
    assert hazard["dual_corroborated_valley_water_km2"] == 0.4560
    assert hazard["ai_only_candidate_water_km2"] == 2.8931
    assert hazard["sar_new_open_water_km2"] == 0.0
    assert hazard["permanent_water_km2"] == 0.0091

    # Settlements list for table and map
    settlements = data["settlements"]
    assert len(settlements) == 505
    cut_count = sum(1 for s in settlements if s["cut"])
    assert cut_count == 49
    sample = settlements[0]
    assert "id" in sample
    assert "n" in sample
    assert "cut" in sample
    assert "status_label" in sample
    assert "x" in sample and "y" in sample

    # Layers map
    layers = data["layers"]
    assert "flood_evidence" in layers
    assert "affected_roads" in layers
    assert "cut_off_settlements" in layers
    assert "settlement_connectivity" in layers
    assert "downstream_path" in layers
    assert "downstream_settlements" in layers
    assert "agreement_map" in layers
    assert "review_full_aoi_png" in layers

    # Downstream summary
    assert "downstream_summary" in data and data["downstream_summary"] is not None
    assert data["downstream_summary"]["path_length_km"] == 50.01
    assert data["downstream_summary"]["settlements_count"] == 20

    # Scientific disclaimers
    assert len(data["scientific_disclaimers"]) >= 4
    assert any("EXPLORATORY AI" in d for d in data["scientific_disclaimers"])
    assert any("HAZARD CORRIDOR" in d for d in data["scientific_disclaimers"])

    # Defensive SVG mock coordinates
    assert data["river"] is not None
    assert data["floodW"] == 26
    assert data["town"] == [640, 190]
    assert data["hospital"] == [560, 300]

    # Test /api/v1/assess alias
    v1_resp = client.post("/api/v1/assess", json=payload)
    assert v1_resp.status_code == 200
    assert v1_resp.json()["summary"]["buildings_affected"] == 873


def test_post_assess_bounding_box_supported(client):
    """Verify POST /api/assess accepts coordinates within Trishuli corridor bounding box."""
    payload = {
        "area": "85.05, 27.85, 85.45, 28.25",
        "date": "2026-08-26",
    }
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 200
    assert resp.json()["summary"]["settlements_cut_off"] == 49


def test_post_assess_unsupported_area_rejected(client):
    """Verify POST /api/assess rejects areas outside verified coverage with clear HTTP 400."""
    payload = {
        "area": "Miami, Florida",
        "date": "2026-08-26",
    }
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 400
    err = resp.json()["detail"]
    assert "outside the supported coverage area" in err
    assert "Trishuli" in err


def test_post_assess_unsupported_date_rejected(client):
    """Verify POST /api/assess rejects dates outside verified event window with clear HTTP 400."""
    payload = {
        "area": "Trishuli, Nepal",
        "date": "2025-01-01",
    }
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 400
    err = resp.json()["detail"]
    assert "outside the verified Trishuli flood event window" in err
    assert "2026-08-26" in err


def test_post_assess_invalid_date_format_rejected(client):
    """Verify POST /api/assess rejects malformed date strings with clear HTTP 400."""
    payload = {
        "area": "Trishuli, Nepal",
        "date": "not-a-valid-date",
    }
    resp = client.post("/api/assess", json=payload)
    assert resp.status_code == 400
    assert "Invalid date format" in resp.json()["detail"]


def test_post_assess_layer_endpoints_work(client):
    """Verify that layers referenced in assessment response can be resolved and retrieved."""
    # Test affected roads
    r_resp = client.get("/api/layers/affected_roads")
    assert r_resp.status_code == 200
    assert len(r_resp.content) > 0

    # Test flood evidence
    f_resp = client.get("/api/layers/flood_evidence")
    assert f_resp.status_code == 200
    assert len(f_resp.content) > 0

    # Test affected bridges
    b_resp = client.get("/api/layers/affected_bridges")
    assert b_resp.status_code == 200
    assert len(b_resp.content) > 0

    # Test settlement connectivity
    s_resp = client.get("/api/layers/settlement_connectivity")
    assert s_resp.status_code == 200
    assert len(s_resp.content) > 0

    # Test downstream path
    d_resp = client.get("/api/layers/downstream_path")
    assert d_resp.status_code == 200
    assert len(d_resp.content) > 0

    # Test downstream settlements
    ds_resp = client.get("/api/layers/downstream_settlements")
    assert ds_resp.status_code == 200
    assert len(ds_resp.content) > 0

    # Test exploratory AI review maps
    p_resp = client.get("/api/layers/review_visualizations_full_aoi")
    assert p_resp.status_code == 200
    assert p_resp.headers["content-type"] == "image/png"

    o_resp = client.get("/api/layers/map_06_evidence_overlap_and_disagreement")
    assert o_resp.status_code == 200
    assert o_resp.headers["content-type"] == "image/png"


def test_frontend_basemap_configuration():
    """Verify frontend map configuration uses public no-key basemaps and does not rely on Carto key."""
    frontend_map_js = Path("frontend/js/ui/map.js")
    if not frontend_map_js.is_file():
        frontend_map_js = Path("C:/Users/user/Downloads/floodintel/floodintel/js/ui/map.js")
    assert frontend_map_js.is_file(), "Frontend map.js must exist"
    code = frontend_map_js.read_text(encoding="utf-8")

    # Genuinely public basemap without API key requirement
    assert "tile.openstreetmap.org" in code, "Must use public OpenStreetMap tiles"
    assert "basemaps.cartocdn.com" not in code, "Must not use deprecated Carto endpoint requiring API key"
    assert "server.arcgisonline.com" in code, "Must use public Esri World Imagery satellite layer"

    # Verify attribution prefix is set cleanly without unstyled inline flag SVG
    assert "setPrefix" in code, "Must configure clean attribution prefix"

    # Verify CSS scoping
    frontend_css = Path("frontend/css/components.css")
    if not frontend_css.is_file():
        frontend_css = Path("C:/Users/user/Downloads/floodintel/floodintel/css/components.css")
    assert frontend_css.is_file(), "Frontend components.css must exist"
    css = frontend_css.read_text(encoding="utf-8")
    assert ".map-wrap > svg#map" in css, "SVG scaling rule must be scoped to #map only"


def test_agreement_map_layer_endpoint(client):
    """Verify validation toolkit agreement map is exposed and served as PNG."""
    resp = client.get("/api/layers/agreement_map")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 1000


def test_validation_toolkit_artifacts_presence_and_isolation():
    """Verify validation reference datasets are strictly confined to data/validation_only."""
    val_only = Path("data/validation_only")
    assert (val_only / "iwm" / "IWM_2026_08_26_Nepal_Flood_VAP02.shp").is_file()
    assert (val_only / "tasa" / "Affected_0816_0828.shp").is_file()
    assert (val_only / "boundaries" / "geoBoundaries-NPL-ADM0.geojson").is_file()

    results_dir = Path("data/runs/validation_trishuli_toolkit")
    assert (results_dir / "validation_results.json").is_file()
    assert (results_dir / "agreement_map.png").is_file()
    assert (results_dir / "self_test_results.json").is_file()



