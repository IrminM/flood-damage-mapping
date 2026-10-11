"""Provenance log: which inputs and which processing produced a result.

Every analysis run writes provenance.json next to its outputs, recording:
  * inputs: path, SHA-256, size, declared source type, role (before/after/dem/osm...), metadata
  * steps: name, parameters, status, duration, outputs, notes
  * environment: python + key package versions, git commit if available
  * data_mode: REAL or SYNTHETIC_TEST (dev_mocks)

This is the evidence for "no prohibited inputs were used" and for reproducing
the Trishuli case study.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .contracts import DataMode

_TRACKED_PACKAGES = ("numpy", "scipy", "rasterio", "pyproj", "shapely", "geopandas",
                     "networkx", "torch", "pydantic", "fastapi")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _environment() -> dict[str, Any]:
    pkgs = {}
    for name in _TRACKED_PACKAGES:
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip() or None
    except Exception:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": pkgs, "git_commit": commit}


class ProvenanceLog:
    def __init__(self, analysis_id: Optional[str] = None, data_mode: DataMode = DataMode.REAL):
        self.analysis_id = analysis_id or uuid.uuid4().hex[:12]
        self.data_mode = data_mode
        self.created = _utcnow()
        self.inputs: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.warnings: list[str] = []

    def add_input(
        self,
        path: Path | str,
        source_type: str,
        role: str,
        metadata: Optional[dict[str, Any]] = None,
        compute_hash: bool = True,
    ) -> dict[str, Any]:
        p = Path(path)
        rec: dict[str, Any] = {"path": str(p), "role": role, "source_type": str(source_type),
                               "metadata": metadata or {}}
        if p.is_file():
            rec["size_bytes"] = p.stat().st_size
            if compute_hash:
                rec["sha256"] = sha256_file(p)
        if source_type == DataMode.SYNTHETIC_TEST.value or source_type == "synthetic-test":
            self.data_mode = DataMode.SYNTHETIC_TEST
        self.inputs.append(rec)
        return rec

    @contextmanager
    def step(self, name: str, **params: Any) -> Iterator[dict[str, Any]]:
        """Record a processing step. Usage: `with prov.step("change_detection", threshold=-3) as s: ...`"""
        rec: dict[str, Any] = {"name": name, "params": params, "started": _utcnow(),
                               "status": "running", "outputs": [], "notes": []}
        self.steps.append(rec)
        t0 = time.perf_counter()
        try:
            yield rec
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc(limit=5)
            raise
        finally:
            rec["duration_s"] = round(time.perf_counter() - t0, 3)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "data_mode": self.data_mode.value,
            "created": self.created,
            "environment": _environment(),
            "inputs": self.inputs,
            "steps": self.steps,
            "notes": self.notes,
            "warnings": self.warnings,
        }

    def write_json(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p
