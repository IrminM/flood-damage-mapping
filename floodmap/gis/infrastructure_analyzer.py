"""Modular Stage-4 Infrastructure Impact Analysis Layer.

Integrates:
  1. Pre-event OpenStreetMap infrastructure (buildings, roads, bridges)
  2. Verified Multimodal Evidence Fusion (dual AI+SAR, AI-only, SAR-only, unvalidated)

Pipeline Architecture:
  OSMLoader
      ↓
  InfrastructureNormalizer
      ↓
  EvidenceSpatialIndex
      ↓
  BuildingImpactAnalyzer | RoadImpactAnalyzer | BridgeImpactAnalyzer
      ↓
  Structured InfrastructureImpactResult
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from ..core.config import ImpactConfig, OSMConfig
from ..core.contracts import (
    BridgeImpactStats,
    BuildingImpactStats,
    DataMode,
    InfrastructureSummary,
    RoadImpactStats,
    SourceType,
    WaterwayContextStats,
)
from ..core.data_rules import DataRulesGuard, to_date
from ..core.errors import InputDataError
from ..data_sources.base import OSMData, OSMSnapshotMeta
from ..data_sources.osm_local import LocalOSMProvider


DISCLAIMER_TEXT = (
    "Infrastructure impact represents spatial intersection with remote-sensing flood and surface change evidence. "
    "Satellite imagery alone does NOT prove structural building destruction or road washout without on-ground inspection. "
    "Water passing beneath an intact bridge deck cannot be distinguished from structural failure. "
    "Features are classified as 'potentially affected' with explicit evidence provenance."
)

EVIDENCE_TIER_LABELS = {
    "high_confidence": "confirmed_dual_evidence",
    "ai_only": "ai_only_candidate",
    "sar_only": "sar_only_evidence",
    "unvalidated": "unvalidated_hillside_change",
    "baseline_water": "permanent_water_baseline",
}


# ==============================================================================
# 1. OSM Loader
# ==============================================================================

class OSMLoader:
    """Loads and validates pre-event OpenStreetMap snapshots."""

    def __init__(self, guard: Optional[DataRulesGuard] = None, cfg: Optional[OSMConfig] = None):
        self.guard = guard
        self.cfg = cfg or OSMConfig()

    def load_from_dir(self, osm_dir: Path | str, event_date: date) -> OSMData:
        """Load pre-event OSM layers from directory with date check."""
        osm_path = Path(osm_dir)
        if not osm_path.is_dir():
            raise InputDataError(f"OSM directory not found: {osm_path}")

        meta_path = osm_path / "snapshot_meta.json"
        if not meta_path.exists():
            raise InputDataError(
                f"{meta_path} missing: pre-event snapshot date verification is mandatory"
            )

        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        snapshot_date = to_date(raw_meta.get("snapshot_date"))
        if self.guard:
            self.guard.check_osm_snapshot(snapshot_date, event_date)
        elif snapshot_date >= event_date:
            raise InputDataError(
                f"Post-event OSM data is prohibited: snapshot {snapshot_date} >= event {event_date}"
            )

        if self.guard:
            provider = LocalOSMProvider(self.guard, self.cfg)
            return provider.load(osm_path, event_date)

        # Fallback direct loader if guard is omitted
        layers: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for lyr in ("buildings", "roads", "bridges", "places", "health"):
            lyr_p = osm_path / f"{lyr}.geojson"
            if lyr_p.exists():
                layers[lyr] = json.loads(lyr_p.read_text(encoding="utf-8"))
            else:
                missing.append(lyr)
                layers[lyr] = {"type": "FeatureCollection", "features": []}

        meta = OSMSnapshotMeta(
            snapshot_date=snapshot_date,
            source=str(raw_meta.get("source", "unknown")),
            bbox=tuple(raw_meta["bbox"]) if raw_meta.get("bbox") else None,
            fetched_at=raw_meta.get("fetched_at"),
            filters=raw_meta.get("filters", {}),
        )
        return OSMData(meta=meta, layers=layers, source_dir=osm_path, missing_layers=missing)


# ==============================================================================
# 2. Infrastructure Normalizer
# ==============================================================================

class InfrastructureNormalizer:
    """Cleans geometries, assigns canonical identifiers, and handles CRS projection."""

    def __init__(self, target_crs: str = "EPSG:32645"):
        self.target_crs = target_crs

    def normalize(
        self,
        fc_or_gdf: dict[str, Any] | gpd.GeoDataFrame,
        layer_name: str,
        src_crs: str = "EPSG:4326",
    ) -> gpd.GeoDataFrame:
        """Clean geometries, ensure IDs, and project to metric target CRS."""
        if isinstance(fc_or_gdf, gpd.GeoDataFrame):
            gdf = fc_or_gdf.copy()
        else:
            features = fc_or_gdf.get("features", [])
            if not features:
                return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry", crs=self.target_crs)
            gdf = gpd.GeoDataFrame.from_features(features, crs=src_crs)
            # Extract ID from feature top-level 'id' or properties
            feat_ids = []
            for i, feat in enumerate(features):
                fid = feat.get("id")
                if fid in (None, ""):
                    props = feat.get("properties") or {}
                    fid = props.get("@osmId") or props.get("osm_id") or props.get("id") or f"{layer_name}/{i}"
                feat_ids.append(str(fid))
            gdf["id"] = feat_ids

        if gdf.empty:
            return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry", crs=self.target_crs)

        # Ensure ID column has no missing values
        if "id" not in gdf.columns:
            if "@osmId" in gdf.columns:
                gdf["id"] = gdf["@osmId"].astype(str)
            elif "osm_id" in gdf.columns:
                gdf["id"] = gdf["osm_id"].astype(str)
            else:
                gdf["id"] = [f"{layer_name}/{i}" for i in range(len(gdf))]
        else:
            gdf["id"] = gdf["id"].fillna("").astype(str)
            missing_ids = gdf["id"] == ""
            if missing_ids.any():
                gdf.loc[missing_ids, "id"] = [f"{layer_name}/{i}" for i in np.where(missing_ids)[0]]

        # Remove invalid / empty geometries
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry", crs=self.target_crs)

        # Make valid geometries
        gdf.geometry = gdf.geometry.map(lambda g: make_valid(g) if not g.is_valid else g)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

        # Reproject to target metric CRS
        if gdf.crs is None:
            raise InputDataError(f"Layer {layer_name} has no CRS defined")
        if gdf.crs != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)

        return gdf


# ==============================================================================
# 3. Evidence Spatial Index
# ==============================================================================

class EvidenceSpatialIndex:
    """Builds an R-tree spatial index over target flood and change evidence polygons."""

    def __init__(
        self,
        evidence_source: gpd.GeoDataFrame | Path | str,
        target_crs: str = "EPSG:32645",
        allowed_categories: Optional[Sequence[str]] = None,
    ):
        self.target_crs = target_crs
        if isinstance(evidence_source, (str, Path)):
            self.evidence_gdf = gpd.read_file(evidence_source)
        else:
            self.evidence_gdf = evidence_source.copy()

        if self.evidence_gdf.crs is None:
            raise InputDataError("Evidence layer must have a defined CRS")
        if self.evidence_gdf.crs != self.target_crs:
            self.evidence_gdf = self.evidence_gdf.to_crs(self.target_crs)

        # Standardize columns
        if "category_name" not in self.evidence_gdf.columns and "class_name" in self.evidence_gdf.columns:
            self.evidence_gdf["category_name"] = self.evidence_gdf["class_name"]
        if "strength_name" not in self.evidence_gdf.columns and "evidence_name" in self.evidence_gdf.columns:
            self.evidence_gdf["strength_name"] = self.evidence_gdf["evidence_name"]

        # Default allowed categories (exclude background and terrain_rejected)
        if allowed_categories is None:
            self.allowed_categories = {
                "ai_water_and_sar_valley_change",
                "ai_and_sar_flood",
                "ai_only_water",
                "sar_only_valley_change",
                "sar_only_hillside_change",
                "permanent_water",
            }
        else:
            self.allowed_categories = set(allowed_categories)

        if not self.evidence_gdf.empty and "category_name" in self.evidence_gdf.columns:
            self.filtered_gdf = self.evidence_gdf[
                self.evidence_gdf["category_name"].isin(self.allowed_categories)
            ].copy()
        else:
            self.filtered_gdf = self.evidence_gdf.copy()

        # Clean geometries
        if not self.filtered_gdf.empty:
            self.filtered_gdf.geometry = self.filtered_gdf.geometry.map(lambda g: make_valid(g) if not g.is_valid else g)
            self.filtered_gdf = self.filtered_gdf[~self.filtered_gdf.geometry.is_empty].copy()

    @property
    def is_empty(self) -> bool:
        return self.filtered_gdf.empty

    @property
    def sindex(self):
        return self.filtered_gdf.sindex


# ==============================================================================
# 4. Building Impact Analyzer
# ==============================================================================

class BuildingImpactAnalyzer:
    """Analyzes building footprints intersecting detected flood and change evidence."""

    def __init__(self, config: Optional[ImpactConfig] = None):
        self.cfg = config or ImpactConfig()

    def analyze(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        evidence_index: EvidenceSpatialIndex,
    ) -> Tuple[gpd.GeoDataFrame, BuildingImpactStats]:
        b = buildings_gdf.copy()
        if b.empty:
            return b, BuildingImpactStats()

        b["footprint_area_m2"] = b.geometry.area.round(2)
        b["affected_area_m2"] = 0.0
        b["affected_fraction"] = 0.0
        b["is_affected"] = False
        b["impact_class"] = "none"
        b["evidence_source"] = "none"
        b["evidence_tier"] = "none"
        b["structural_damage_claimed"] = False
        b["impact_description"] = "Clear; no detected intersection with flood or change evidence"

        if evidence_index.is_empty:
            stats = BuildingImpactStats(
                total_count=len(b),
                total_footprint_m2=round(float(b["footprint_area_m2"].sum()), 2),
            )
            return b, stats

        ev = evidence_index.filtered_gdf
        eval_geom = b.geometry.buffer(self.cfg.building_buffer_m) if self.cfg.building_buffer_m > 0 else b.geometry
        b_eval = gpd.GeoDataFrame({"b_idx": b.index}, geometry=eval_geom, crs=b.crs)

        joined = gpd.sjoin(b_eval, ev, how="inner", predicate="intersects")
        if not joined.empty:
            for b_idx, group in joined.groupby("b_idx"):
                b_poly = b_eval.loc[b_idx, "geometry"]
                total_intersected_m2 = 0.0
                class_areas: dict[str, float] = {}
                strengths: list[str] = []

                for ev_idx in group["index_right"]:
                    ev_poly = ev.loc[ev_idx, "geometry"]
                    c_name = str(ev.loc[ev_idx, "category_name"])
                    s_name = str(ev.loc[ev_idx, "strength_name"])
                    inter = b_poly.intersection(ev_poly)
                    if not inter.is_empty:
                        a = float(inter.area)
                        total_intersected_m2 += a
                        class_areas[c_name] = class_areas.get(c_name, 0.0) + a
                        strengths.append(s_name)

                if total_intersected_m2 >= self.cfg.min_affected_building_area_m2:
                    orig_a = max(float(b.loc[b_idx, "footprint_area_m2"]), 0.01)
                    dom_class = max(class_areas.items(), key=lambda kv: kv[1])[0] if class_areas else "none"

                    # Hierarchy for dominant evidence tier
                    if "high_confidence" in strengths or "confirmed_by_both" in strengths:
                        dom_tier = "confirmed_dual_evidence"
                        dom_ev = "high_confidence"
                    elif "ai_only" in strengths:
                        dom_tier = "ai_only_candidate"
                        dom_ev = "ai_only"
                    elif "sar_only" in strengths:
                        dom_tier = "sar_only_evidence"
                        dom_ev = "sar_only"
                    elif "unvalidated" in strengths:
                        dom_tier = "unvalidated_hillside_change"
                        dom_ev = "unvalidated"
                    elif "baseline_water" in strengths:
                        dom_tier = "permanent_water_baseline"
                        dom_ev = "baseline_water"
                    else:
                        dom_tier = "candidate_evidence"
                        dom_ev = strengths[0] if strengths else "none"

                    b.loc[b_idx, "is_affected"] = True
                    b.loc[b_idx, "affected_area_m2"] = round(total_intersected_m2, 2)
                    b.loc[b_idx, "affected_fraction"] = round(min(1.0, total_intersected_m2 / orig_a), 4)
                    b.loc[b_idx, "impact_class"] = dom_class
                    b.loc[b_idx, "evidence_source"] = dom_ev
                    b.loc[b_idx, "evidence_tier"] = dom_tier
                    b.loc[b_idx, "structural_damage_claimed"] = False
                    b.loc[b_idx, "impact_description"] = (
                        f"Potentially inundated ({dom_class}, {dom_tier}); "
                        "satellite evidence does NOT prove structural damage."
                    )

        aff_mask = b["is_affected"]
        total_count = len(b)
        aff_count = int(aff_mask.sum())
        total_fp = float(b["footprint_area_m2"].sum())
        aff_fp = float(b.loc[aff_mask, "affected_area_m2"].sum())

        by_class: dict[str, int] = {}
        by_ev: dict[str, int] = {}
        if aff_count > 0:
            by_class = b.loc[aff_mask, "impact_class"].value_counts().to_dict()
            by_ev = b.loc[aff_mask, "evidence_tier"].value_counts().to_dict()

        stats = BuildingImpactStats(
            total_count=total_count,
            potentially_affected_count=aff_count,
            affected_pct=round(100.0 * aff_count / max(total_count, 1), 2),
            total_footprint_m2=round(total_fp, 2),
            affected_footprint_m2=round(aff_fp, 2),
            by_class=by_class,
            by_evidence=by_ev,
        )
        return b, stats


# ==============================================================================
# 5. Road Impact Analyzer
# ==============================================================================

class RoadImpactAnalyzer:
    """Analyzes road centerlines intersecting detected flood and change evidence."""

    def __init__(self, config: Optional[ImpactConfig] = None):
        self.cfg = config or ImpactConfig()

    def analyze(
        self,
        roads_gdf: gpd.GeoDataFrame,
        evidence_index: EvidenceSpatialIndex,
    ) -> Tuple[gpd.GeoDataFrame, RoadImpactStats]:
        r = roads_gdf.copy()
        if r.empty:
            return r, RoadImpactStats()

        r["total_length_m"] = r.geometry.length.round(2)
        r["total_length_km"] = (r["total_length_m"] / 1000.0).round(4)
        r["affected_length_m"] = 0.0
        r["affected_length_km"] = 0.0
        r["affected_fraction"] = 0.0
        r["is_affected"] = False
        r["impact_class"] = "none"
        r["evidence_source"] = "none"
        r["evidence_tier"] = "none"
        r["structural_damage_claimed"] = False
        r["impact_description"] = "Clear; no detected intersection with flood or change evidence"

        if evidence_index.is_empty:
            total_km = round(float(r["total_length_m"].sum()) / 1000.0, 3)
            stats = RoadImpactStats(total_length_km=total_km, total_segments=len(r))
            return r, stats

        ev = evidence_index.filtered_gdf
        ev_eval_geom = ev.geometry.buffer(self.cfg.road_buffer_m) if self.cfg.road_buffer_m > 0 else ev.geometry
        ev_eval = gpd.GeoDataFrame(
            {"ev_idx": ev.index, "category_name": ev["category_name"], "strength_name": ev["strength_name"]},
            geometry=ev_eval_geom,
            crs=ev.crs,
        )

        joined = gpd.sjoin(r, ev_eval, how="inner", predicate="intersects")
        if not joined.empty:
            for r_idx, group in joined.groupby(joined.index):
                r_line = r.loc[r_idx, "geometry"]
                total_intersected_m = 0.0
                class_lengths: dict[str, float] = {}
                strengths: list[str] = []

                for ev_idx in group["ev_idx"]:
                    ev_poly = ev_eval.loc[ev_idx, "geometry"]
                    c_name = str(ev_eval.loc[ev_idx, "category_name"])
                    s_name = str(ev_eval.loc[ev_idx, "strength_name"])
                    inter = r_line.intersection(ev_poly)
                    if not inter.is_empty:
                        l = float(inter.length)
                        total_intersected_m += l
                        class_lengths[c_name] = class_lengths.get(c_name, 0.0) + l
                        strengths.append(s_name)

                if total_intersected_m >= self.cfg.min_affected_road_length_m:
                    tot_l = max(float(r.loc[r_idx, "total_length_m"]), 0.01)
                    dom_class = max(class_lengths.items(), key=lambda kv: kv[1])[0] if class_lengths else "none"

                    if "high_confidence" in strengths or "confirmed_by_both" in strengths:
                        dom_tier = "confirmed_dual_evidence"
                        dom_ev = "high_confidence"
                    elif "ai_only" in strengths:
                        dom_tier = "ai_only_candidate"
                        dom_ev = "ai_only"
                    elif "sar_only" in strengths:
                        dom_tier = "sar_only_evidence"
                        dom_ev = "sar_only"
                    elif "unvalidated" in strengths:
                        dom_tier = "unvalidated_hillside_change"
                        dom_ev = "unvalidated"
                    elif "baseline_water" in strengths:
                        dom_tier = "permanent_water_baseline"
                        dom_ev = "baseline_water"
                    else:
                        dom_tier = "candidate_evidence"
                        dom_ev = strengths[0] if strengths else "none"

                    r.loc[r_idx, "is_affected"] = True
                    r.loc[r_idx, "affected_length_m"] = round(total_intersected_m, 2)
                    r.loc[r_idx, "affected_length_km"] = round(total_intersected_m / 1000.0, 4)
                    r.loc[r_idx, "affected_fraction"] = round(min(1.0, total_intersected_m / tot_l), 4)
                    r.loc[r_idx, "impact_class"] = dom_class
                    r.loc[r_idx, "evidence_source"] = dom_ev
                    r.loc[r_idx, "evidence_tier"] = dom_tier
                    r.loc[r_idx, "structural_damage_claimed"] = False
                    r.loc[r_idx, "impact_description"] = (
                        f"Potentially flooded ({dom_class}, {dom_tier}); "
                        "satellite evidence does NOT prove road washout or structural destruction."
                    )

        aff_mask = r["is_affected"]
        total_len_km = float(r["total_length_m"].sum()) / 1000.0
        aff_len_km = float(r.loc[aff_mask, "affected_length_m"].sum()) / 1000.0

        by_hwy: dict[str, dict[str, float]] = {}
        if "highway" in r.columns:
            for hwy, group in r.groupby("highway"):
                hwy_str = str(hwy)
                t_km = float(group["total_length_m"].sum()) / 1000.0
                a_km = float(group.loc[group["is_affected"], "affected_length_m"].sum()) / 1000.0
                by_hwy[hwy_str] = {"total_km": round(t_km, 3), "affected_km": round(a_km, 3)}

        by_class: dict[str, float] = {}
        by_ev: dict[str, float] = {}
        if aff_mask.any():
            for c_name, group in r.loc[aff_mask].groupby("impact_class"):
                by_class[str(c_name)] = round(float(group["affected_length_m"].sum()) / 1000.0, 3)
            for ev_name, group in r.loc[aff_mask].groupby("evidence_tier"):
                by_ev[str(ev_name)] = round(float(group["affected_length_m"].sum()) / 1000.0, 3)

        stats = RoadImpactStats(
            total_length_km=round(total_len_km, 3),
            potentially_affected_length_km=round(aff_len_km, 3),
            affected_pct=round(100.0 * aff_len_km / max(total_len_km, 1e-4), 2),
            total_segments=len(r),
            potentially_affected_segments=int(aff_mask.sum()),
            by_highway_type_km=by_hwy,
            by_class_km=by_class,
            by_evidence_km=by_ev,
        )
        return r, stats


# ==============================================================================
# 6. Bridge Impact Analyzer
# ==============================================================================

class BridgeImpactAnalyzer:
    """Analyzes bridge features with spatial buffer proximity to flood/change evidence."""

    def __init__(self, config: Optional[ImpactConfig] = None):
        self.cfg = config or ImpactConfig()

    def analyze(
        self,
        bridges_gdf: gpd.GeoDataFrame,
        evidence_index: EvidenceSpatialIndex,
    ) -> Tuple[gpd.GeoDataFrame, BridgeImpactStats]:
        br = bridges_gdf.copy()
        if br.empty:
            return br, BridgeImpactStats()

        br["is_affected"] = False
        br["impact_class"] = "none"
        br["evidence_source"] = "none"
        br["evidence_tier"] = "none"
        br["structural_damage_claimed"] = False
        br["impact_description"] = "Clear; no detected intersection with flood or change evidence"

        if evidence_index.is_empty:
            return br, BridgeImpactStats(total_count=len(br))

        ev = evidence_index.filtered_gdf
        eval_geom = br.geometry.buffer(self.cfg.bridge_buffer_m) if self.cfg.bridge_buffer_m > 0 else br.geometry
        br_eval = gpd.GeoDataFrame({"br_idx": br.index}, geometry=eval_geom, crs=br.crs)

        joined = gpd.sjoin(br_eval, ev, how="inner", predicate="intersects")
        if not joined.empty:
            for br_idx, group in joined.groupby("br_idx"):
                c_names = ev.loc[group["index_right"], "category_name"].tolist()
                s_names = ev.loc[group["index_right"], "strength_name"].tolist()

                dom_class = "ai_water_and_sar_valley_change" if "ai_water_and_sar_valley_change" in c_names else c_names[0]
                if "high_confidence" in s_names or "confirmed_by_both" in s_names:
                    dom_tier = "confirmed_dual_evidence"
                    dom_ev = "high_confidence"
                elif "ai_only" in s_names:
                    dom_tier = "ai_only_candidate"
                    dom_ev = "ai_only"
                elif "sar_only" in s_names:
                    dom_tier = "sar_only_evidence"
                    dom_ev = "sar_only"
                elif "unvalidated" in s_names:
                    dom_tier = "unvalidated_hillside_change"
                    dom_ev = "unvalidated"
                elif "baseline_water" in s_names:
                    dom_tier = "permanent_water_baseline"
                    dom_ev = "baseline_water"
                else:
                    dom_tier = "candidate_evidence"
                    dom_ev = s_names[0] if s_names else "none"

                br.loc[br_idx, "is_affected"] = True
                br.loc[br_idx, "impact_class"] = dom_class
                br.loc[br_idx, "evidence_source"] = dom_ev
                br.loc[br_idx, "evidence_tier"] = dom_tier
                br.loc[br_idx, "structural_damage_claimed"] = False
                br.loc[br_idx, "impact_description"] = (
                    f"Potentially affected / approach flooded ({dom_class}, {dom_tier}); "
                    "water passing beneath an intact bridge deck cannot be distinguished from structural failure."
                )

        aff_mask = br["is_affected"]
        total_count = len(br)
        aff_count = int(aff_mask.sum())

        by_class: dict[str, int] = {}
        by_ev: dict[str, int] = {}
        if aff_count > 0:
            by_class = br.loc[aff_mask, "impact_class"].value_counts().to_dict()
            by_ev = br.loc[aff_mask, "evidence_tier"].value_counts().to_dict()

        stats = BridgeImpactStats(
            total_count=total_count,
            potentially_affected_count=aff_count,
            affected_pct=round(100.0 * aff_count / max(total_count, 1), 2),
            by_class=by_class,
            by_evidence=by_ev,
        )
        return br, stats


# ==============================================================================
# 7. Structured Infrastructure Impact Result
# ==============================================================================

@dataclass
class InfrastructureImpactResult:
    """Consolidated outcome of Stage-4 Infrastructure Impact Assessment."""
    buildings_gdf: gpd.GeoDataFrame
    roads_gdf: gpd.GeoDataFrame
    bridges_gdf: gpd.GeoDataFrame
    building_stats: BuildingImpactStats
    road_stats: RoadImpactStats
    bridge_stats: BridgeImpactStats
    summary: InfrastructureSummary
    impact_dict: dict[str, Any]
    provenance: dict[str, Any]

    def save_outputs(self, out_dir: Path | str) -> dict[str, Path]:
        """Save all 6 required Stage 4 output files."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        # 1. infrastructure_impact.json
        impact_path = out_dir / "infrastructure_impact.json"
        impact_path.write_text(json.dumps(self.impact_dict, indent=2), encoding="utf-8")
        outputs["infrastructure_impact"] = impact_path

        # 2. affected_buildings.geojson
        b_path = out_dir / "affected_buildings.geojson"
        self.buildings_gdf.to_crs("EPSG:4326").to_file(b_path, driver="GeoJSON")
        outputs["affected_buildings"] = b_path

        # 3. affected_roads.geojson
        r_path = out_dir / "affected_roads.geojson"
        self.roads_gdf.to_crs("EPSG:4326").to_file(r_path, driver="GeoJSON")
        outputs["affected_roads"] = r_path

        # 4. affected_bridges.geojson
        br_path = out_dir / "affected_bridges.geojson"
        self.bridges_gdf.to_crs("EPSG:4326").to_file(br_path, driver="GeoJSON")
        outputs["affected_bridges"] = br_path

        # 5. infrastructure_summary.json
        sum_path = out_dir / "infrastructure_summary.json"
        sum_path.write_text(self.summary.model_dump_json(indent=2), encoding="utf-8")
        outputs["infrastructure_summary"] = sum_path

        # 6. provenance.json
        prov_path = out_dir / "provenance.json"
        prov_path.write_text(json.dumps(self.provenance, indent=2), encoding="utf-8")
        outputs["provenance"] = prov_path

        return outputs


# ==============================================================================
# 8. High-Level Coordinator Function
# ==============================================================================

def analyze_infrastructure_impact(
    osm_source: OSMData | Path | str,
    evidence_source: gpd.GeoDataFrame | Path | str,
    event_date: date,
    target_crs: str = "EPSG:32645",
    config: Optional[ImpactConfig] = None,
    guard: Optional[DataRulesGuard] = None,
    analysis_id: Optional[str] = None,
    data_mode: DataMode = DataMode.REAL,
    aoi_name: Optional[str] = None,
) -> InfrastructureImpactResult:
    """Execute complete modular Stage-4 infrastructure impact analysis."""
    analysis_id = analysis_id or uuid.uuid4().hex[:12]
    cfg = config or ImpactConfig()

    # Step 1: OSM Loader
    loader = OSMLoader(guard=guard)
    if isinstance(osm_source, (str, Path)):
        osm_data = loader.load_from_dir(osm_source, event_date)
        osm_dir_str = str(osm_source)
    elif isinstance(osm_source, OSMData):
        osm_data = osm_source
        if guard:
            guard.check_osm_snapshot(osm_data.meta.snapshot_date, event_date)
        osm_dir_str = str(osm_data.source_dir) if osm_data.source_dir else "in_memory"
    else:
        raise InputDataError(f"Invalid osm_source type: {type(osm_source)}")

    # Step 2: Infrastructure Normalizer
    normalizer = InfrastructureNormalizer(target_crs=target_crs)
    b_raw = normalizer.normalize(osm_data.layers.get("buildings", {}), "buildings")
    r_raw = normalizer.normalize(osm_data.layers.get("roads", {}), "roads")
    br_raw = normalizer.normalize(osm_data.layers.get("bridges", {}), "bridges")

    # Step 3: Evidence Spatial Index
    ev_index = EvidenceSpatialIndex(evidence_source, target_crs=target_crs)

    # Step 4: Analyzers
    b_analyzer = BuildingImpactAnalyzer(config=cfg)
    b_analyzed, b_stats = b_analyzer.analyze(b_raw, ev_index)

    r_analyzer = RoadImpactAnalyzer(config=cfg)
    r_analyzed, r_stats = r_analyzer.analyze(r_raw, ev_index)

    br_analyzer = BridgeImpactAnalyzer(config=cfg)
    br_analyzed, br_stats = br_analyzer.analyze(br_raw, ev_index)

    # Step 5: Summary and Provenance
    summary = InfrastructureSummary(
        analysis_id=analysis_id,
        data_mode=data_mode,
        event_date=event_date,
        aoi_name=aoi_name,
        osm_snapshot_date=osm_data.meta.snapshot_date,
        crs=target_crs,
        buildings=b_stats,
        roads=r_stats,
        bridges=br_stats,
        parameters=cfg.model_dump(),
        outputs={
            "infrastructure_impact": "infrastructure_impact.json",
            "affected_buildings": "affected_buildings.geojson",
            "affected_roads": "affected_roads.geojson",
            "affected_bridges": "affected_bridges.geojson",
            "infrastructure_summary": "infrastructure_summary.json",
        },
        warnings=list(osm_data.warnings),
        interpretation_notes=[DISCLAIMER_TEXT],
    )

    impact_dict = {
        "analysis_id": analysis_id,
        "event_date": event_date.isoformat(),
        "osm_snapshot_date": osm_data.meta.snapshot_date.isoformat(),
        "osm_source": osm_data.meta.source,
        "target_crs": target_crs,
        "metrics_summary": {
            "buildings": b_stats.model_dump(),
            "roads": r_stats.model_dump(),
            "bridges": br_stats.model_dump(),
        },
        "scientific_disclaimer": DISCLAIMER_TEXT,
    }

    provenance = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_id": analysis_id,
        "pipeline_stage": "Stage 4 (Infrastructure Impact Assessment)",
        "inputs": {
            "osm_pre_event": {
                "source_dir": osm_dir_str,
                "snapshot_date": osm_data.meta.snapshot_date.isoformat(),
                "source": osm_data.meta.source,
                "feature_counts": {
                    "buildings": len(b_raw),
                    "roads": len(r_raw),
                    "bridges": len(br_raw),
                },
            },
            "evidence_layer": {
                "source": str(evidence_source) if isinstance(evidence_source, (str, Path)) else "in_memory",
                "polygon_count": len(ev_index.filtered_gdf),
                "crs": target_crs,
            },
        },
        "parameters": cfg.model_dump(),
        "scientific_principles": {
            "structural_damage_claimed": False,
            "disclaimer": DISCLAIMER_TEXT,
        },
    }

    return InfrastructureImpactResult(
        buildings_gdf=b_analyzed,
        roads_gdf=r_analyzed,
        bridges_gdf=br_analyzed,
        building_stats=b_stats,
        road_stats=r_stats,
        bridge_stats=br_stats,
        summary=summary,
        impact_dict=impact_dict,
        provenance=provenance,
    )
