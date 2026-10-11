"""Vectorize raster flood and change evidence into geospatial vector layers.

Converts canonical raster classes (Class 1 flood water, Class 3 surface change)
into shapely geometries and GeoPandas GeoDataFrames with full class and evidence attribution.
"""
from __future__ import annotations

from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely.validation import make_valid

from ..core.contracts import EvidenceSource, OutputClass
from ..core.georaster import GeoRaster

CLASS_NAMES = {
    int(OutputClass.BACKGROUND): "background",
    int(OutputClass.FLOOD_WATER): "flood_water",
    int(OutputClass.PERMANENT_WATER): "permanent_water",
    int(OutputClass.DEBRIS_OR_CHANGE): "debris_or_change",
    int(OutputClass.NODATA): "nodata",
}

EVIDENCE_NAMES = {
    int(EvidenceSource.NONE): "none",
    int(EvidenceSource.CHANGE_DETECTION): "change_detection_only",
    int(EvidenceSource.AI_SEGMENTATION): "ai_segmentation_only",
    int(EvidenceSource.BOTH): "confirmed_by_both",
}


def vectorize_evidence(
    classes_raster: GeoRaster,
    evidence_raster: Optional[GeoRaster] = None,
    target_classes: Sequence[int] = (int(OutputClass.FLOOD_WATER), int(OutputClass.DEBRIS_OR_CHANGE)),
) -> gpd.GeoDataFrame:
    """Vectorize target classes from a raster into a GeoDataFrame.

    Parameters:
        classes_raster: GeoRaster containing canonical OutputClass values.
        evidence_raster: Optional GeoRaster containing EvidenceSource flags.
        target_classes: Classes to extract as polygons (default: flood water and debris).

    Returns:
        GeoDataFrame with columns: geometry, class_code, class_name, evidence_source, evidence_name.
    """
    crs = classes_raster.crs
    transform = classes_raster.transform
    c_data = classes_raster.data[0]

    # Mask of observable target pixels
    target_set = set(target_classes)
    mask = np.isin(c_data, list(target_set)) & (c_data != int(OutputClass.NODATA))

    columns = ["geometry", "class_code", "class_name", "evidence_source", "evidence_name"]
    if not mask.any():
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)

    if evidence_raster is not None:
        ev_data = evidence_raster.data[0]
        # Encode combined value: class * 10 + evidence
        coded = (c_data.astype(np.int32) * 10 + ev_data.astype(np.int32)).astype(np.int32)
    else:
        coded = c_data.astype(np.int32)

    geoms = []
    class_codes = []
    class_names = []
    ev_sources = []
    ev_names = []

    for geom_dict, val in rasterio.features.shapes(coded, mask=mask, transform=transform):
        val_int = int(val)
        if evidence_raster is not None:
            c_code = val_int // 10
            ev_code = val_int % 10
        else:
            c_code = val_int
            ev_code = int(EvidenceSource.NONE)

        g = shape(geom_dict)
        if not g.is_valid:
            g = make_valid(g)
        if g.is_empty:
            continue

        geoms.append(g)
        class_codes.append(c_code)
        class_names.append(CLASS_NAMES.get(c_code, f"class_{c_code}"))
        ev_sources.append(ev_code)
        ev_names.append(EVIDENCE_NAMES.get(ev_code, f"evidence_{ev_code}"))

    if not geoms:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)

    gdf = gpd.GeoDataFrame(
        {
            "class_code": class_codes,
            "class_name": class_names,
            "evidence_source": ev_sources,
            "evidence_name": ev_names,
        },
        geometry=geoms,
        crs=crs,
    )
    return gdf
