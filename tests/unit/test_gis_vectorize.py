"""Unit tests for raster vectorization (floodmap.gis.vectorize)."""
import numpy as np
import pytest
import rasterio.transform

from floodmap.core.contracts import EvidenceSource, OutputClass
from floodmap.core.georaster import GeoRaster
from floodmap.gis.vectorize import vectorize_evidence


@pytest.fixture
def sample_classes_raster():
    # 20x20 pixels, 10m resolution, origin (1000, 2000)
    # Background=0, Flood=1 in [5:10, 5:10], Debris=3 in [12:15, 12:15]
    H, W = 20, 20
    data = np.zeros((1, H, W), dtype=np.uint8)
    data[0, 5:10, 5:10] = int(OutputClass.FLOOD_WATER)
    data[0, 12:15, 12:15] = int(OutputClass.DEBRIS_OR_CHANGE)
    transform = (10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    return GeoRaster(data=data, transform=transform, crs="EPSG:32645", nodata=255)


@pytest.fixture
def sample_evidence_raster():
    H, W = 20, 20
    data = np.zeros((1, H, W), dtype=np.uint8)
    # Flood pixels at [5:10, 5:10] have evidence=BOTH (3)
    data[0, 5:10, 5:10] = int(EvidenceSource.BOTH)
    # Debris pixels at [12:15, 12:15] have evidence=CHANGE_DETECTION (1)
    data[0, 12:15, 12:15] = int(EvidenceSource.CHANGE_DETECTION)
    transform = (10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    return GeoRaster(data=data, transform=transform, crs="EPSG:32645", nodata=0)


def test_vectorize_evidence_basic(sample_classes_raster, sample_evidence_raster):
    gdf = vectorize_evidence(sample_classes_raster, sample_evidence_raster)

    assert not gdf.empty
    assert gdf.crs == "EPSG:32645"
    assert len(gdf) == 2  # one flood polygon, one debris polygon

    flood_row = gdf[gdf["class_code"] == int(OutputClass.FLOOD_WATER)].iloc[0]
    assert flood_row["class_name"] == "flood_water"
    assert flood_row["evidence_source"] == int(EvidenceSource.BOTH)
    assert flood_row["evidence_name"] == "confirmed_by_both"
    # 5x5 pixels = 25 pixels * 100 m² = 2500 m²
    assert flood_row.geometry.area == pytest.approx(2500.0)

    debris_row = gdf[gdf["class_code"] == int(OutputClass.DEBRIS_OR_CHANGE)].iloc[0]
    assert debris_row["class_name"] == "debris_or_change"
    assert debris_row["evidence_source"] == int(EvidenceSource.CHANGE_DETECTION)
    # 3x3 pixels = 9 pixels * 100 m² = 900 m²
    assert debris_row.geometry.area == pytest.approx(900.0)


def test_vectorize_empty_when_no_targets(sample_classes_raster):
    # Only background raster
    data = np.zeros((1, 10, 10), dtype=np.uint8)
    r = GeoRaster(data=data, transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0), crs="EPSG:32645")
    gdf = vectorize_evidence(r)
    assert gdf.empty
    assert "geometry" in gdf.columns
    assert gdf.crs == "EPSG:32645"
