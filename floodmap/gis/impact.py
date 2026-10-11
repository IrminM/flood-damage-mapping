"""Spatial impact assessment of flood and change evidence against OSM infrastructure.

Evaluates spatial intersections of pre-event buildings, roads, bridges, and waterways
with strictly parameterized thresholds, metric calculations in raster CRS,
and safe NoData/invalid geometry handling.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from ..core.config import ImpactConfig
from ..core.contracts import (
    BridgeImpactStats,
    BuildingImpactStats,
    EvidenceSource,
    OutputClass,
    RoadImpactStats,
    WaterwayContextStats,
)
from ..data_sources.base import OSMData


def _to_gdf(fc: dict[str, Any], src_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Safely convert a GeoJSON FeatureCollection dict into a GeoDataFrame."""
    features = fc.get("features", [])
    if not features:
        return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry", crs=src_crs)

    gdf = gpd.GeoDataFrame.from_features(features, crs=src_crs)
    if "id" not in gdf.columns:
        gdf["id"] = [f.get("id", f"feature_{i}") for i, f in enumerate(features)]

    # Validate geometries
    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
    gdf = gdf[valid_mask].copy()
    if not gdf.empty:
        gdf.geometry = gdf.geometry.map(lambda g: make_valid(g) if not g.is_valid else g)
    return gdf


import pyproj


def _resolve_metric_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> str:
    """Ensure a projected metric CRS is used for metric measurements (meters, m²)."""
    crs = pyproj.CRS(target_crs)
    if crs.is_geographic:
        try:
            estimated = gdf.estimate_utm_crs()
            if estimated:
                return str(estimated)
        except Exception:
            pass
        return "EPSG:3857"
    return target_crs


def assess_buildings(
    buildings_fc: dict[str, Any],
    evidence_gdf: gpd.GeoDataFrame,
    target_crs: str,
    cfg: ImpactConfig,
) -> Tuple[gpd.GeoDataFrame, BuildingImpactStats]:
    """Assess flood impact on buildings."""
    b_gdf = _to_gdf(buildings_fc, src_crs="EPSG:4326")
    if b_gdf.empty:
        return b_gdf, BuildingImpactStats()

    metric_crs = _resolve_metric_crs(b_gdf, target_crs)
    b_metric = b_gdf.to_crs(metric_crs)
    ev_metric = evidence_gdf.to_crs(metric_crs) if (not evidence_gdf.empty and evidence_gdf.crs != metric_crs) else evidence_gdf

    b_metric["footprint_area_m2"] = b_metric.geometry.area.round(2)
    b_metric["affected_area_m2"] = 0.0
    b_metric["affected_fraction"] = 0.0
    b_metric["is_affected"] = False
    b_metric["impact_class"] = "none"
    b_metric["evidence_source"] = "none"

    if evidence_gdf.empty:
        stats = BuildingImpactStats(
            total_count=len(b_metric),
            total_footprint_m2=round(float(b_metric["footprint_area_m2"].sum()), 2),
        )
        return b_metric, stats

    # Buffer buildings if configured
    eval_geom = b_metric.geometry
    if cfg.building_buffer_m > 0:
        eval_geom = eval_geom.buffer(cfg.building_buffer_m)

    # Spatial intersection using sindex
    b_eval_gdf = gpd.GeoDataFrame(
        {"b_idx": b_metric.index, "orig_area": b_metric["footprint_area_m2"]},
        geometry=eval_geom,
        crs=metric_crs,
    )

    joined = gpd.sjoin(b_eval_gdf, ev_metric, how="inner", predicate="intersects")
    if not joined.empty:
        for b_idx, group in joined.groupby("b_idx"):
            b_poly = b_eval_gdf.loc[b_idx, "geometry"]
            total_intersected_m2 = 0.0
            class_areas: dict[str, float] = {}
            ev_sources: list[str] = []

            for ev_idx in group["index_right"]:
                ev_poly = ev_metric.loc[ev_idx, "geometry"]
                c_name = ev_metric.loc[ev_idx, "class_name"]
                ev_name = ev_metric.loc[ev_idx, "evidence_name"]

                inter = b_poly.intersection(ev_poly)
                if not inter.is_empty:
                    area = inter.area
                    total_intersected_m2 += area
                    class_areas[c_name] = class_areas.get(c_name, 0.0) + area
                    ev_sources.append(ev_name)

            if total_intersected_m2 >= cfg.min_affected_building_area_m2:
                orig_a = max(b_metric.loc[b_idx, "footprint_area_m2"], 0.01)
                dom_class = max(class_areas.items(), key=lambda kv: kv[1])[0] if class_areas else "none"
                dom_ev = "confirmed_by_both" if "confirmed_by_both" in ev_sources else (
                    ev_sources[0] if ev_sources else "none"
                )

                b_metric.loc[b_idx, "is_affected"] = True
                b_metric.loc[b_idx, "affected_area_m2"] = round(float(total_intersected_m2), 2)
                b_metric.loc[b_idx, "affected_fraction"] = round(float(min(1.0, total_intersected_m2 / orig_a)), 4)
                b_metric.loc[b_idx, "impact_class"] = dom_class
                b_metric.loc[b_idx, "evidence_source"] = dom_ev

    affected_mask = b_metric["is_affected"]
    total_count = len(b_metric)
    aff_count = int(affected_mask.sum())
    total_fp = float(b_metric["footprint_area_m2"].sum())
    aff_fp = float(b_metric.loc[affected_mask, "affected_area_m2"].sum())

    by_class: dict[str, int] = {}
    by_ev: dict[str, int] = {}
    if aff_count > 0:
        by_class = b_metric.loc[affected_mask, "impact_class"].value_counts().to_dict()
        by_ev = b_metric.loc[affected_mask, "evidence_source"].value_counts().to_dict()

    stats = BuildingImpactStats(
        total_count=total_count,
        potentially_affected_count=aff_count,
        affected_pct=round(100.0 * aff_count / max(total_count, 1), 2),
        total_footprint_m2=round(total_fp, 2),
        affected_footprint_m2=round(aff_fp, 2),
        by_class=by_class,
        by_evidence=by_ev,
    )
    return b_metric, stats


def assess_roads(
    roads_fc: dict[str, Any],
    evidence_gdf: gpd.GeoDataFrame,
    target_crs: str,
    cfg: ImpactConfig,
) -> Tuple[gpd.GeoDataFrame, RoadImpactStats]:
    """Assess flood impact on individual road segments."""
    r_gdf = _to_gdf(roads_fc, src_crs="EPSG:4326")
    if r_gdf.empty:
        return r_gdf, RoadImpactStats()

    metric_crs = _resolve_metric_crs(r_gdf, target_crs)
    r_metric = r_gdf.to_crs(metric_crs)
    ev_metric = evidence_gdf.to_crs(metric_crs) if (not evidence_gdf.empty and evidence_gdf.crs != metric_crs) else evidence_gdf

    r_metric["total_length_m"] = r_metric.geometry.length.round(2)
    r_metric["affected_length_m"] = 0.0
    r_metric["affected_fraction"] = 0.0
    r_metric["is_affected"] = False
    r_metric["impact_class"] = "none"
    r_metric["evidence_source"] = "none"

    if ev_metric.empty:
        stats = RoadImpactStats(
            total_length_km=round(float(r_metric["total_length_m"].sum()) / 1000.0, 3),
            total_segments=len(r_metric),
        )
        return r_metric, stats

    # Dilate flood evidence by road buffer tolerance if configured
    ev_eval_geom = ev_metric.geometry
    if cfg.road_buffer_m > 0:
        ev_eval_geom = ev_eval_geom.buffer(cfg.road_buffer_m)

    ev_eval_gdf = gpd.GeoDataFrame(
        {"ev_idx": ev_metric.index, "class_name": ev_metric["class_name"], "evidence_name": ev_metric["evidence_name"]},
        geometry=ev_eval_geom,
        crs=metric_crs,
    )

    joined = gpd.sjoin(r_metric, ev_eval_gdf, how="inner", predicate="intersects")
    if not joined.empty:
        for r_idx, group in joined.groupby(joined.index):
            r_line = r_metric.loc[r_idx, "geometry"]
            total_intersected_m = 0.0
            class_lengths: dict[str, float] = {}
            ev_sources: list[str] = []

            for ev_idx in group["ev_idx"]:
                ev_poly = ev_eval_gdf.loc[ev_idx, "geometry"]
                c_name = ev_eval_gdf.loc[ev_idx, "class_name"]
                ev_name = ev_eval_gdf.loc[ev_idx, "evidence_name"]

                inter = r_line.intersection(ev_poly)
                if not inter.is_empty:
                    length = inter.length
                    total_intersected_m += length
                    class_lengths[c_name] = class_lengths.get(c_name, 0.0) + length
                    ev_sources.append(ev_name)

            if total_intersected_m >= cfg.min_affected_road_length_m:
                total_len = max(r_metric.loc[r_idx, "total_length_m"], 0.01)
                dom_class = max(class_lengths.items(), key=lambda kv: kv[1])[0] if class_lengths else "none"
                dom_ev = "confirmed_by_both" if "confirmed_by_both" in ev_sources else (
                    ev_sources[0] if ev_sources else "none"
                )

                r_metric.loc[r_idx, "is_affected"] = True
                r_metric.loc[r_idx, "affected_length_m"] = round(float(total_intersected_m), 2)
                r_metric.loc[r_idx, "affected_fraction"] = round(float(min(1.0, total_intersected_m / total_len)), 4)
                r_metric.loc[r_idx, "impact_class"] = dom_class
                r_metric.loc[r_idx, "evidence_source"] = dom_ev

    affected_mask = r_metric["is_affected"]
    total_len_km = float(r_metric["total_length_m"].sum()) / 1000.0
    aff_len_km = float(r_metric.loc[affected_mask, "affected_length_m"].sum()) / 1000.0

    by_hwy: dict[str, dict[str, float]] = {}
    if "highway" in r_metric.columns:
        for hwy, group in r_metric.groupby("highway"):
            hwy_str = str(hwy)
            t_km = float(group["total_length_m"].sum()) / 1000.0
            a_km = float(group.loc[group["is_affected"], "affected_length_m"].sum()) / 1000.0
            by_hwy[hwy_str] = {"total_km": round(t_km, 3), "affected_km": round(a_km, 3)}

    by_class: dict[str, float] = {}
    by_ev: dict[str, float] = {}
    if affected_mask.any():
        for c_name, group in r_metric.loc[affected_mask].groupby("impact_class"):
            by_class[str(c_name)] = round(float(group["affected_length_m"].sum()) / 1000.0, 3)
        for ev_name, group in r_metric.loc[affected_mask].groupby("evidence_source"):
            by_ev[str(ev_name)] = round(float(group["affected_length_m"].sum()) / 1000.0, 3)

    stats = RoadImpactStats(
        total_length_km=round(total_len_km, 3),
        potentially_affected_length_km=round(aff_len_km, 3),
        affected_pct=round(100.0 * aff_len_km / max(total_len_km, 1e-4), 2),
        total_segments=len(r_metric),
        potentially_affected_segments=int(affected_mask.sum()),
        by_highway_type_km=by_hwy,
        by_class_km=by_class,
        by_evidence_km=by_ev,
    )
    return r_metric, stats


def assess_bridges(
    bridges_fc: dict[str, Any],
    evidence_gdf: gpd.GeoDataFrame,
    target_crs: str,
    cfg: ImpactConfig,
) -> Tuple[gpd.GeoDataFrame, BridgeImpactStats]:
    """Assess flood and change evidence proximity to bridges."""
    br_gdf = _to_gdf(bridges_fc, src_crs="EPSG:4326")
    if br_gdf.empty:
        return br_gdf, BridgeImpactStats()

    metric_crs = _resolve_metric_crs(br_gdf, target_crs)
    br_metric = br_gdf.to_crs(metric_crs)
    ev_metric = evidence_gdf.to_crs(metric_crs) if (not evidence_gdf.empty and evidence_gdf.crs != metric_crs) else evidence_gdf

    br_metric["is_affected"] = False
    br_metric["impact_class"] = "none"
    br_metric["evidence_source"] = "none"

    if ev_metric.empty:
        stats = BridgeImpactStats(total_count=len(br_metric))
        return br_metric, stats

    # Buffer bridges by spatial tolerance (default 15m)
    eval_geom = br_metric.geometry
    if cfg.bridge_buffer_m > 0:
        eval_geom = eval_geom.buffer(cfg.bridge_buffer_m)

    br_eval_gdf = gpd.GeoDataFrame({"br_idx": br_metric.index}, geometry=eval_geom, crs=metric_crs)
    joined = gpd.sjoin(br_eval_gdf, ev_metric, how="inner", predicate="intersects")
    if not joined.empty:
        for br_idx, group in joined.groupby("br_idx"):
            br_metric.loc[br_idx, "is_affected"] = True
            c_names = ev_metric.loc[group["index_right"], "class_name"].tolist()
            ev_names = ev_metric.loc[group["index_right"], "evidence_name"].tolist()

            dom_class = "flood_water" if "flood_water" in c_names else c_names[0]
            dom_ev = "confirmed_by_both" if "confirmed_by_both" in ev_names else ev_names[0]

            br_metric.loc[br_idx, "impact_class"] = dom_class
            br_metric.loc[br_idx, "evidence_source"] = dom_ev

    aff_mask = br_metric["is_affected"]
    total_count = len(br_metric)
    aff_count = int(aff_mask.sum())

    by_class: dict[str, int] = {}
    by_ev: dict[str, int] = {}
    if aff_count > 0:
        by_class = br_metric.loc[aff_mask, "impact_class"].value_counts().to_dict()
        by_ev = br_metric.loc[aff_mask, "evidence_source"].value_counts().to_dict()

    stats = BridgeImpactStats(
        total_count=total_count,
        potentially_affected_count=aff_count,
        affected_pct=round(100.0 * aff_count / max(total_count, 1), 2),
        by_class=by_class,
        by_evidence=by_ev,
    )
    return br_metric, stats


def assess_waterway_context(
    waterways_fc: Optional[dict[str, Any]],
    evidence_gdf: gpd.GeoDataFrame,
    target_crs: str,
    cfg: ImpactConfig,
) -> Optional[WaterwayContextStats]:
    """Assess whether Class 3 non-water surface change lies near pre-event river channels."""
    if not waterways_fc or not waterways_fc.get("features"):
        return None

    w_gdf = _to_gdf(waterways_fc, src_crs="EPSG:4326")
    if w_gdf.empty:
        return None

    metric_crs = _resolve_metric_crs(w_gdf, target_crs)
    w_metric = w_gdf.to_crs(metric_crs)
    ev_metric = evidence_gdf.to_crs(metric_crs) if (not evidence_gdf.empty and evidence_gdf.crs != metric_crs) else evidence_gdf

    total_km = round(float(w_metric.geometry.length.sum()) / 1000.0, 3)

    # Class 3 polygons (non-water surface change)
    class3_gdf = ev_metric[ev_metric["class_code"] == int(OutputClass.DEBRIS_OR_CHANGE)]
    if class3_gdf.empty:
        return WaterwayContextStats(
            total_waterways_km=total_km,
            class3_near_waterway_km2=0.0,
            class3_away_from_waterway_km2=0.0,
            buffer_distance_m=cfg.waterway_buffer_m,
        )

    # Buffer waterways
    buffered = w_metric.geometry.buffer(cfg.waterway_buffer_m)
    channel_buffer = buffered.union_all() if hasattr(buffered, "union_all") else buffered.unary_union
    near_m2 = 0.0
    total_c3_m2 = 0.0

    for poly in class3_gdf.geometry:
        total_c3_m2 += poly.area
        inter = poly.intersection(channel_buffer)
        if not inter.is_empty:
            near_m2 += inter.area

    near_km2 = round(near_m2 / 1e6, 4)
    away_km2 = round(max(0.0, total_c3_m2 - near_m2) / 1e6, 4)

    return WaterwayContextStats(
        total_waterways_km=total_km,
        class3_near_waterway_km2=near_km2,
        class3_away_from_waterway_km2=away_km2,
        buffer_distance_m=cfg.waterway_buffer_m,
    )

