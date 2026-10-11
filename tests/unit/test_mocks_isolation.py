"""Guarantees that development mocks cannot leak into the real pipeline."""
import ast
from pathlib import Path

import numpy as np

from dev_mocks.synthetic_scene import make_synthetic_s1_pair
from floodmap.core.contracts import DataMode, SourceType

ROOT = Path(__file__).resolve().parents[2]


def test_floodmap_never_imports_dev_mocks():
    offenders = []
    for py in (ROOT / "floodmap").rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "dev_mocks" or n.startswith("dev_mocks.") for n in names):
                offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"floodmap/ must not import dev_mocks: {offenders}"


def test_synthetic_pair_is_labelled_and_has_signal():
    p = make_synthetic_s1_pair(seed=1)
    for m in (p.before_meta, p.after_meta):
        assert m.data_mode is DataMode.SYNTHETIC_TEST and m.source_type is SourceType.SYNTHETIC_TEST
    assert p.before.metadata["data_mode"] == "SYNTHETIC_TEST"
    assert p.before.same_grid(p.after)
    vv_b, vv_a = p.before.band("VV"), p.after.band("VV")
    # flooded pixels get much darker; untouched land does not
    land = ~(p.flood_truth | p.permanent_water | p.change_truth | p.slope_false_water | p.after_nodata)
    assert np.median(vv_a[p.flood_truth]) < 0.2 * np.median(vv_b[p.flood_truth])
    assert 0.8 < np.median(vv_a[land]) / np.median(vv_b[land]) < 1.25
    # surface-change patch brightens (~+6 dB) but is not water
    assert np.median(vv_a[p.change_truth]) > 3 * np.median(vv_b[p.change_truth])
    assert np.isnan(vv_a[p.after_nodata]).all()
    # DEM: floodplain is low and flat, valley sides rise
    assert p.dem.band_names == ["elevation"] and p.dem.res == (30.0, 30.0)
    z = p.dem.band("elevation")
    assert z[:, z.shape[1] // 2].mean() < z[:, 0].mean() - 100
