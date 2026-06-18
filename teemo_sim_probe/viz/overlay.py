"""Mask + label overlay on the RGB frame (matplotlib, no cv2 dependency)."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np

from ..core.schema import Graph
from ..core.mask_extractor import MaskAccumulator


def _color_for(node_id: str) -> np.ndarray:
    h = hashlib.md5(node_id.encode()).digest()
    rgb = np.array([h[0], h[1], h[2]], dtype=float) / 255.0
    # brighten
    return 0.4 + 0.6 * rgb


def render_overlay(
    rgb: np.ndarray,
    graph: Graph,
    masks: MaskAccumulator,
    out_path: str,
    alpha: float = 0.5,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = rgb.shape[:2]
    canvas = rgb.astype(float) / 255.0
    blended = canvas.copy()

    for node in graph.nodes:
        m = masks.get(node.node_id)
        if m is None or m.sum() == 0:
            continue
        color = _color_for(node.node_id)
        blended[m] = (1 - alpha) * blended[m] + alpha * color

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(np.clip(blended, 0, 1))
    ax.axis("off")

    # Label each node at its mask centroid.
    for node in graph.nodes:
        m = masks.get(node.node_id)
        if m is None or m.sum() == 0:
            continue
        ys, xs = np.nonzero(m)
        cx, cy = float(xs.mean()), float(ys.mean())
        tag = "ee" if node.node_type == "ee" else node.name
        ax.text(
            cx, cy, tag,
            color="white", fontsize=9, ha="center", va="center",
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path
