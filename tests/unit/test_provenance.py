import hashlib
import json

import pytest

from floodmap.core.contracts import DataMode
from floodmap.core.provenance import ProvenanceLog


def test_inputs_hashed_and_steps_timed(tmp_path):
    f = tmp_path / "in.bin"
    f.write_bytes(b"hello")
    prov = ProvenanceLog(analysis_id="abc")
    rec = prov.add_input(f, "sentinel-1", role="before", metadata={"orbit": 85})
    assert rec["sha256"] == hashlib.sha256(b"hello").hexdigest() and rec["size_bytes"] == 5

    with prov.step("change_detection", threshold_db=-3.0) as s:
        s["outputs"].append("change.tif")
    with pytest.raises(RuntimeError):
        with prov.step("broken"):
            raise RuntimeError("boom")

    out = prov.write_json(tmp_path / "run" / "provenance.json")
    d = json.loads(out.read_text())
    assert d["analysis_id"] == "abc" and d["data_mode"] == "REAL"
    assert d["steps"][0]["status"] == "ok" and d["steps"][0]["params"] == {"threshold_db": -3.0}
    assert d["steps"][1]["status"] == "failed" and "boom" in d["steps"][1]["error"]
    assert "python" in d["environment"]


def test_synthetic_input_marks_run(tmp_path):
    prov = ProvenanceLog()
    prov.add_input(tmp_path / "missing.tif", "synthetic-test", role="before")
    assert prov.data_mode is DataMode.SYNTHETIC_TEST
