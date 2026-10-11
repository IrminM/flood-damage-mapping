"""ohsome API client for *pre-event* OpenStreetMap snapshots.

ohsome (https://api.ohsome.org/v1) serves OSM history, so we can request the map
exactly as it was on a date before the event (challenge rule: no post-event edits).

Request building and response normalisation are pure functions (unit-tested offline);
only `fetch_layer` performs network I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.config import OSMConfig
from ..core.contracts import AOI, SourceType
from ..core.data_rules import DataRulesGuard
from ..core.geo_utils import aoi_bounds, buffer_lonlat_bounds_km

Bounds = tuple[float, float, float, float]

# Layers that need the wide (network) extent vs only the AOI.
_NETWORK_LAYERS = {"roads", "bridges", "places", "health"}


@dataclass
class OhsomeRequest:
    layer: str
    url: str
    params: dict[str, str]
    bbox: Bounds


def _fmt_bbox(b: Bounds) -> str:
    return ",".join(f"{v:.6f}" for v in b)


def build_requests(aoi: AOI, snapshot_date: date, cfg: OSMConfig) -> list[OhsomeRequest]:
    """One POST request per layer. Roads/bridges/places/health use AOI + network buffer."""
    aoi_b = aoi_bounds(aoi)
    net_b = buffer_lonlat_bounds_km(aoi_b, cfg.network_buffer_km)
    bld_b = buffer_lonlat_bounds_km(aoi_b, cfg.buildings_buffer_km)
    url = cfg.ohsome_url.rstrip("/") + "/elements/geometry"
    reqs = []
    for layer in cfg.layers:
        if layer not in cfg.filters:
            raise ValueError(f"No ohsome filter configured for layer {layer!r} (config osm.filters)")
        bbox = net_b if layer in _NETWORK_LAYERS else bld_b
        reqs.append(OhsomeRequest(
            layer=layer, url=url, bbox=bbox,
            params={
                "bboxes": _fmt_bbox(bbox),
                "time": snapshot_date.isoformat(),
                "filter": " ".join(cfg.filters[layer].split()),
                "properties": "tags",
            },
        ))
    return reqs


def normalize_response(layer: str, fc: dict[str, Any]) -> dict[str, Any]:
    """Keep only features with geometry; give each a stable id from '@osmId'."""
    if fc.get("type") != "FeatureCollection":
        raise ValueError(f"ohsome response for {layer} is not a FeatureCollection")
    out = []
    for i, f in enumerate(fc.get("features", [])):
        if not f.get("geometry"):
            continue
        props = f.get("properties") or {}
        out.append({"type": "Feature", "id": str(props.get("@osmId", f"{layer}/{i}")),
                    "geometry": f["geometry"], "properties": props})
    return {"type": "FeatureCollection", "features": out}


def _parquet_to_feature_collection(layer: str, content: bytes) -> dict[str, Any]:
    import io
    import geopandas as gpd
    import shapely.geometry

    gdf = gpd.read_parquet(io.BytesIO(content))
    features = []
    for idx, row in gdf.iterrows():
        geom = getattr(row, "geom", None) or getattr(row, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        tags_raw = getattr(row, "tags", None)
        if isinstance(tags_raw, dict):
            tags = dict(tags_raw)
        elif isinstance(tags_raw, (list, tuple)):
            tags = dict(tags_raw)
        else:
            tags = {}
        osm_id_val = getattr(row, "osm_id", None)
        osm_type_val = getattr(row, "osm_type", None)
        if osm_id_val is not None and osm_type_val is not None:
            full_id = f"{osm_type_val}/{osm_id_val}"
        elif osm_id_val is not None:
            full_id = f"{layer}/{osm_id_val}"
        else:
            full_id = f"{layer}/{idx}"

        tags["@osmId"] = full_id
        tags["osm_id"] = full_id
        features.append({
            "type": "Feature",
            "id": full_id,
            "geometry": shapely.geometry.mapping(geom),
            "properties": tags,
        })
    return {"type": "FeatureCollection", "features": features}


def fetch_layer(
    req: OhsomeRequest,
    post: Optional[Callable[..., Any]] = None,
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Execute one request (network). `post` is injectable for tests."""
    is_network_call = post is None
    if post is None:
        import requests
        post = requests.post

    import os
    token = api_key or os.environ.get("OHSOME_API_KEY") or os.environ.get("HEIGIT_API_KEY")
    headers = {
        "User-Agent": "FloodDamageMapper/1.0 (academic disaster assessment; +https://github.com)",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = token

    try:
        resp = post(req.url, data=req.params, timeout=timeout, headers=headers)
    except TypeError:
        # Fallback for test stubs with fixed signature (url, data, timeout)
        resp = post(req.url, data=req.params, timeout=timeout)

    status_code = getattr(resp, "status_code", None)
    if status_code == 403:
        if token and is_network_call:
            # HeiGIT requires ohsome v2 extraction endpoint for geometry
            try:
                import requests as req_mod
                v2_url = "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet"
                time_iso = req.params["time"]
                if "T" not in time_iso:
                    time_iso = f"{time_iso}T00:00:00Z"
                v2_payload = {
                    "aoi": list(req.bbox),
                    "time": time_iso,
                    "filter": req.params["filter"],
                    "clip": False,
                }
                v2_resp = req_mod.post(
                    v2_url,
                    headers={"authorization": token, "User-Agent": "FloodDamageMapper/1.0"},
                    json=v2_payload,
                    timeout=timeout,
                )
                if v2_resp.status_code == 200:
                    return _parquet_to_feature_collection(req.layer, v2_resp.content)
            except Exception:
                pass

        raise RuntimeError(
            f"ohsome API returned HTTP 403 Forbidden for layer '{req.layer}'. "
            "HeiGIT ohsome API requires an API key for geometry extractions (/elements/geometry). "
            "Please obtain a free API key at https://account.heigit.org/signup and set the OHSOME_API_KEY "
            "environment variable or pass --api-key."
        )

    resp.raise_for_status()
    return normalize_response(req.layer, resp.json())


def fetch_snapshot(
    aoi: AOI,
    event_date: date,
    snapshot_date: date,
    out_dir: Path,
    cfg: OSMConfig,
    guard: DataRulesGuard,
    post: Optional[Callable[..., Any]] = None,
    log: Callable[[str], None] = print,
    api_key: Optional[str] = None,
) -> Path:
    """Fetch all layers and write <out_dir>/<layer>.geojson + snapshot_meta.json."""
    guard.check_osm_snapshot(snapshot_date, event_date)  # refuse post-event snapshots up front
    out_dir.mkdir(parents=True, exist_ok=True)
    reqs = build_requests(aoi, snapshot_date, cfg)
    counts = {}
    for req in reqs:
        log(f"ohsome: {req.layer} @ {snapshot_date} bbox={req.params['bboxes']}")
        fc = fetch_layer(req, post=post, api_key=api_key)
        (out_dir / f"{req.layer}.geojson").write_text(json.dumps(fc), encoding="utf-8")
        counts[req.layer] = len(fc["features"])
        log(f"  -> {counts[req.layer]} features")
    meta = {
        "snapshot_date": snapshot_date.isoformat(),
        "event_date": event_date.isoformat(),
        "source": "ohsome API (OpenStreetMap history)",
        "source_type": SourceType.OSM_PRE_EVENT.value,
        "api_url": cfg.ohsome_url,
        "bbox": list(aoi_bounds(aoi)),
        "layer_bboxes": {r.layer: list(r.bbox) for r in reqs},
        "filters": {r.layer: r.params["filter"] for r in reqs},
        "feature_counts": counts,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attribution": "© OpenStreetMap contributors (ODbL)",
    }
    (out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir
