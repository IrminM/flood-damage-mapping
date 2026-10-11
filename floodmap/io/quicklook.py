"""PNG quicklooks for eyeballing stage outputs (not analysis products).

Uses matplotlib with the non-interactive Agg backend. Synthetic runs are watermarked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

CLASS_COLORS = {0: "#e6e6e6", 1: "#2b83ba", 2: "#08306b", 3: "#f28e2b"}
CLASS_NAMES = {0: "no detected change", 1: "new open water", 2: "water in both dates",
               3: "non-water backscatter change (not necessarily debris)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, ax, title: str, path: Path, watermark: Optional[str]) -> Path:
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if watermark:
        ax.text(0.5, 0.5, watermark, transform=ax.transAxes, ha="center", va="center",
                fontsize=18, color="red", alpha=0.35, rotation=20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    _plt().close(fig)
    return path


def save_db_image(arr_db: np.ndarray, path: Path, title: str, vmin: float = -25, vmax: float = 0,
                  watermark: Optional[str] = None) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6 * arr_db.shape[0] / max(arr_db.shape[1], 1) + 0.6))
    im = ax.imshow(np.ma.masked_invalid(arr_db), cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="dB")
    return _finish(fig, ax, title, path, watermark)


def save_log_ratio(lr_db: np.ndarray, path: Path, title: str, lim: float = 10, watermark: Optional[str] = None) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6 * lr_db.shape[0] / max(lr_db.shape[1], 1) + 0.6))
    im = ax.imshow(np.ma.masked_invalid(lr_db), cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="after − before (dB)")
    return _finish(fig, ax, title, path, watermark)


def save_classes(classes: np.ndarray, path: Path, title: str, watermark: Optional[str] = None) -> Path:
    plt = _plt()
    from matplotlib.patches import Patch

    rgb = np.ones(classes.shape + (3,))
    for code, hexcol in CLASS_COLORS.items():
        rgb[classes == code] = [int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    fig, ax = plt.subplots(figsize=(6, 6 * classes.shape[0] / max(classes.shape[1], 1) + 1.2))
    ax.imshow(rgb, interpolation="nearest")
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in (1, 2, 3, 0)]
    handles.append(Patch(facecolor="white", edgecolor="#999", label="not observable / outside AOI"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False)
    return _finish(fig, ax, title, path, watermark)


CONTEXT_COLORS = {0: "#e6e6e6", 1: "#d7301f", 2: "#9e9ac8", 3: "#636363"}
CONTEXT_NAMES = {1: "change in valley bottom (low HAND, gentle slope)", 2: "change on hillside",
                 3: "change, terrain context unknown", 0: "no non-water change"}


def save_change_context(context: np.ndarray, path: Path, title: str, watermark: Optional[str] = None) -> Path:
    plt = _plt()
    from matplotlib.patches import Patch

    rgb = np.ones(context.shape + (3,))
    for code, hexcol in CONTEXT_COLORS.items():
        rgb[context == code] = [int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    fig, ax = plt.subplots(figsize=(6, 6 * context.shape[0] / max(context.shape[1], 1) + 1.2))
    ax.imshow(rgb, interpolation="nearest")
    handles = [Patch(color=CONTEXT_COLORS[c], label=CONTEXT_NAMES[c]) for c in (1, 2, 3, 0)]
    handles.append(Patch(facecolor="white", edgecolor="#999", label="not observable / outside AOI"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False)
    return _finish(fig, ax, title, path, watermark)


EVIDENCE_COLORS = {0: "#e6e6e6", 1: "#2b83ba", 2: "#9970ab", 3: "#008837"}
EVIDENCE_NAMES = {0: "no flood detected", 1: "change detection only", 2: "AI segmentation only",
                  3: "confirmed by both (high confidence)"}


def save_evidence_source(evidence: np.ndarray, path: Path, title: str, watermark: Optional[str] = None) -> Path:
    plt = _plt()
    from matplotlib.patches import Patch

    rgb = np.ones(evidence.shape + (3,))
    for code, hexcol in EVIDENCE_COLORS.items():
        rgb[evidence == code] = [int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    fig, ax = plt.subplots(figsize=(6, 6 * evidence.shape[0] / max(evidence.shape[1], 1) + 1.2))
    ax.imshow(rgb, interpolation="nearest")
    handles = [Patch(color=EVIDENCE_COLORS[c], label=EVIDENCE_NAMES[c]) for c in (3, 1, 2, 0)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False)
    return _finish(fig, ax, title, path, watermark)


def save_probability(prob: np.ndarray, path: Path, title: str, watermark: Optional[str] = None) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6 * prob.shape[0] / max(prob.shape[1], 1) + 0.6))
    im = ax.imshow(np.ma.masked_invalid(prob), cmap="cividis", vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Water Probability")
    return _finish(fig, ax, title, path, watermark)

