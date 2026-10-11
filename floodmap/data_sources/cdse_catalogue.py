"""Copernicus Data Space Ecosystem (CDSE) catalogue search for Sentinel-1 GRD scenes.

Uses the public OData catalogue (no login needed for search):
    https://catalogue.dataspace.copernicus.eu/odata/v1/Products

Returns SceneMetadata-compatible records (relative orbit, pass direction, platform,
time) so floodmap.satellite.pairing can choose a valid same-track before/after pair.
Query building and parsing are pure functions (unit-tested offline).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Optional

from ..core.contracts import AOI, SceneMetadata, SourceType
from ..core.geo_utils import aoi_bounds

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _wkt_polygon(aoi: AOI) -> str:
    x0, y0, x1, y1 = aoi_bounds(aoi)
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


def build_s1_grd_query(aoi: AOI, event_date: date, days_before: int = 30, days_after: int = 21,
                       top: int = 200) -> dict[str, str]:
    """OData query params for Sentinel-1 IW GRD-HD scenes intersecting the AOI around the event."""
    start = datetime.combine(event_date - timedelta(days=days_before), time(0, 0), tzinfo=timezone.utc)
    end = datetime.combine(event_date + timedelta(days=days_after), time(23, 59, 59), tzinfo=timezone.utc)
    flt = (
        "Collection/Name eq 'SENTINEL-1' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{_wkt_polygon(aoi)}') "
        f"and ContentDate/Start gt {_iso(start)} and ContentDate/Start lt {_iso(end)} "
        "and contains(Name,'IW_GRDH')"
    )
    return {"$filter": flt, "$expand": "Attributes", "$orderby": "ContentDate/Start asc", "$top": str(top)}


def _attrs(product: dict[str, Any]) -> dict[str, Any]:
    return {a.get("Name"): a.get("Value") for a in product.get("Attributes", []) if a.get("Name")}


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_products(payload: dict[str, Any]) -> list[SceneMetadata]:
    """Convert OData products to SceneMetadata (deduplicating COG/non-COG copies)."""
    scenes: dict[str, SceneMetadata] = {}
    for p in payload.get("value", []):
        name = p.get("Name", "")
        base = name.replace("_COG", "").replace(".SAFE", "")
        attrs = _attrs(p)
        pol = str(attrs.get("polarisationChannels", "")).replace("&", ",").replace(" ", "")
        bands = [b for b in pol.split(",") if b] or ["VV", "VH"]
        serial = str(attrs.get("platformSerialIdentifier", "")).lower()
        direction = str(attrs.get("orbitDirection", "")).lower() or None
        rel = attrs.get("relativeOrbitNumber")
        meta = SceneMetadata(
            sensor="S1",
            platform=f"sentinel-1{serial}" if serial else "sentinel-1",
            product_type="GRD",
            acquisition_time=_parse_time(p["ContentDate"]["Start"]),
            bands=bands,
            units="linear",
            processing="catalogue record (not yet processed)",
            source_type=SourceType.SENTINEL_1,
            relative_orbit=int(rel) if rel is not None else None,
            orbit_direction=direction if direction in ("ascending", "descending") else None,
            product_ids=[base],
            extra={"cdse_id": p.get("Id"), "online": p.get("Online")},
        )
        scenes.setdefault(base, meta)
    return sorted(scenes.values(), key=lambda m: m.acquisition_time)


def search_s1_grd(aoi: AOI, event_date: date, get: Optional[Callable[..., Any]] = None, **kw: Any) -> list[SceneMetadata]:
    """Network call: search the CDSE catalogue. `get` is injectable for tests."""
    if get is None:
        import requests
        get = requests.get
    resp = get(ODATA_URL, params=build_s1_grd_query(aoi, event_date, **kw), timeout=120)
    resp.raise_for_status()
    return parse_products(resp.json())


S2_L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]


def build_s2_l1c_query(aoi: AOI, event_date: date, days_before: int = 30, days_after: int = 21,
                       top: int = 200) -> dict[str, str]:
    """OData query params for Sentinel-2 Level-1C scenes intersecting the AOI around the event."""
    start = datetime.combine(event_date - timedelta(days=days_before), time(0, 0), tzinfo=timezone.utc)
    end = datetime.combine(event_date + timedelta(days=days_after), time(23, 59, 59), tzinfo=timezone.utc)
    flt = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{_wkt_polygon(aoi)}') "
        f"and ContentDate/Start gt {_iso(start)} and ContentDate/Start lt {_iso(end)} "
        "and contains(Name,'MSIL1C')"
    )
    return {"$filter": flt, "$expand": "Attributes", "$orderby": "ContentDate/Start asc", "$top": str(top)}


def parse_s2_products(payload: dict[str, Any]) -> list[SceneMetadata]:
    """Convert OData Sentinel-2 products to SceneMetadata."""
    scenes: dict[str, SceneMetadata] = {}
    for p in payload.get("value", []):
        name = p.get("Name", "")
        base = name.replace(".SAFE", "")
        attrs = _attrs(p)
        serial = str(attrs.get("platformSerialIdentifier", "")).lower()
        cloud = attrs.get("cloudCover")
        cloud_val = float(cloud) if cloud is not None else None
        rel = attrs.get("relativeOrbitNumber")
        meta = SceneMetadata(
            sensor="S2",
            platform=f"sentinel-2{serial}" if serial else "sentinel-2",
            product_type="L1C",
            acquisition_time=_parse_time(p["ContentDate"]["Start"]),
            bands=list(S2_L1C_BANDS),
            units="reflectance",
            processing="catalogue record (not yet processed)",
            source_type=SourceType.SENTINEL_2,
            relative_orbit=int(rel) if rel is not None else None,
            orbit_direction="descending",
            product_ids=[base],
            cloud_cover_pct=cloud_val,
            extra={"cdse_id": p.get("Id"), "online": p.get("Online")},
        )
        scenes.setdefault(base, meta)
    return sorted(scenes.values(), key=lambda m: m.acquisition_time)


def search_s2_l1c(aoi: AOI, event_date: date, get: Optional[Callable[..., Any]] = None, **kw: Any) -> list[SceneMetadata]:
    """Network call: search the CDSE catalogue for Sentinel-2 L1C scenes."""
    if get is None:
        import requests
        get = requests.get
    resp = get(ODATA_URL, params=build_s2_l1c_query(aoi, event_date, **kw), timeout=120)
    resp.raise_for_status()
    return parse_s2_products(resp.json())

