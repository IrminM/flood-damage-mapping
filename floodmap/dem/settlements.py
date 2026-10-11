"""Pre-event OSM settlement identification and ordering along downstream flood paths."""
from __future__ import annotations

from typing import Any, Optional

import geopandas as gpd
from shapely.geometry import Point

from ..core.config import DownstreamConfig
from ..core.contracts import DownstreamSettlementRecord
from .tracer import DownstreamPathResult


def _to_gdf(fc_or_gdf: Any, src_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Normalize GeoJSON FeatureCollection or GeoDataFrame to GeoDataFrame."""
    if isinstance(fc_or_gdf, gpd.GeoDataFrame):
        return fc_or_gdf.copy()
    if not isinstance(fc_or_gdf, dict) or "features" not in fc_or_gdf:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)
    features = fc_or_gdf.get("features", [])
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)
    gdf = gpd.GeoDataFrame.from_features(features, crs=src_crs)
    if "id" not in gdf.columns:
        gdf["id"] = [f.get("id", f"feature_{i}") for i, f in enumerate(features)]
    return gdf


def trace_nearby_settlements(
    path_result: DownstreamPathResult,
    places_fc: Any,
    cfg: DownstreamConfig,
) -> tuple[gpd.GeoDataFrame, list[DownstreamSettlementRecord]]:
    """Identify and order pre-event settlements situated along or near the downstream path.

    - Evaluates distance in metric CRS against settlement_proximity_radius_m.
    - Projects each settlement point onto the downstream path centerline to establish
      its deterministic along-path distance from the upstream starting point.
    - Preserves stable OSM identifiers and original properties.
    """
    places_gdf = _to_gdf(places_fc, src_crs="EPSG:4326")
    if places_gdf.empty:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty_gdf, []

    # Filter by configured settlement place types if 'place' column is present
    target_types = set(cfg.settlement_place_types)
    if "place" in places_gdf.columns:
        valid_mask = places_gdf["place"].astype(str).str.lower().isin(target_types) | places_gdf["place"].isna()
        eval_gdf = places_gdf[valid_mask].copy()
    else:
        eval_gdf = places_gdf.copy()

    if eval_gdf.empty:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty_gdf, []

    line_metric = path_result.line_geom_metric
    if line_metric.is_empty:
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return empty_gdf, []

    # Project settlements to path metric CRS
    eval_metric = eval_gdf.to_crs(path_result.metric_crs)

    matched_candidates: list[dict[str, Any]] = []

    for idx, row in eval_metric.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        pt_metric = geom if geom.geom_type == "Point" else geom.centroid
        dist_to_path_m = float(line_metric.distance(pt_metric))

        if dist_to_path_m <= cfg.settlement_proximity_radius_m:
            dist_along_path_m = float(line_metric.project(pt_metric))
            orig_row = eval_gdf.loc[idx]
            orig_geom = orig_row.geometry
            orig_pt = orig_geom if orig_geom.geom_type == "Point" else orig_geom.centroid

            osm_id = str(orig_row.get("osm_id") or orig_row.get("@osmId") or orig_row.get("id") or f"place/{idx}")
            name = orig_row.get("name")
            place_type = str(orig_row.get("place", "settlement"))

            matched_candidates.append({
                "geometry": orig_pt,
                "settlement_id": osm_id,
                "settlement_name": str(name) if name is not None else None,
                "place_type": place_type,
                "distance_to_path_m": round(dist_to_path_m, 1),
                "distance_along_path_m": round(dist_along_path_m, 1),
                "distance_along_path_km": round(dist_along_path_m / 1000.0, 2),
            })

    # Sort deterministically by distance along the path from upstream start
    matched_candidates.sort(key=lambda x: x["distance_along_path_m"])

    records: list[DownstreamSettlementRecord] = []
    gdf_rows: list[dict[str, Any]] = []

    for order_idx, item in enumerate(matched_candidates, start=1):
        rec = DownstreamSettlementRecord(
            order=order_idx,
            settlement_id=item["settlement_id"],
            settlement_name=item["settlement_name"],
            place_type=item["place_type"],
            distance_to_path_m=item["distance_to_path_m"],
            distance_along_path_m=item["distance_along_path_m"],
            distance_along_path_km=item["distance_along_path_km"],
        )
        records.append(rec)

        row_dict = dict(item)
        row_dict["order"] = order_idx
        gdf_rows.append(row_dict)

    if gdf_rows:
        settlements_gdf = gpd.GeoDataFrame(gdf_rows, crs="EPSG:4326")
    else:
        settlements_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return settlements_gdf, records
