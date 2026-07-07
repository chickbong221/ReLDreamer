"""Mask overlay on the RGB/depth backdrop, styled like instance-segmentation
SGG figures: each node a distinct saturated color, mask outlines, and labels
placed at mask centroids with small vertical offsets to reduce overlap.

A shared ColorMap can be passed in so colors stay consistent across frames and
match the graph render.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.entity_identity import display_name
from ..core.schema import Graph
from ..core.mask_extractor import MaskAccumulator
from .palette import ColorMap


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """Boolean outline of a binary mask (mask minus its erosion)."""
    m = mask
    er = np.ones_like(m)
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    er &= m
    return m & ~er


def render_overlay(
    rgb: np.ndarray,
    graph: Graph,
    masks: MaskAccumulator,
    out_path: str,
    alpha: float = 0.55,
    colormap: Optional[ColorMap] = None,
    target_inches: float = 6.0,
    dpi: int = 200,
    label_fontsize: Optional[float] = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = colormap or ColorMap()
    # Assign colors in node order so ee is green, then stable per-object.
    cmap.assign_all(graph.node_ids())

    H, W = rgb.shape[:2]
    blended = rgb.astype(float) / 255.0

    # Paint masks (filled, semi-transparent) then outlines (opaque).
    for node in graph.nodes:
        m = masks.get(node.node_id)
        if m is None or m.sum() == 0:
            continue
        color = np.array(cmap.color(node.node_id))
        blended[m] = (1 - alpha) * blended[m] + alpha * color
        outline = _mask_outline(m)
        blended[outline] = color

    # Figure sized to ``target_inches`` on the long side, preserving aspect.
    # This decouples display size from the (often small) sensor resolution, so a
    # 128x128 sensor still produces a large, legible overlay.
    aspect = H / W
    if W >= H:
        figsize = (target_inches, target_inches * aspect)
    else:
        figsize = (target_inches / aspect, target_inches)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # interpolation="nearest" keeps mask edges crisp when upscaling small images
    ax.imshow(np.clip(blended, 0, 1), interpolation="nearest")
    ax.axis("off")

    # Font scales with figure size so labels stay proportional at any resolution.
    fs = label_fontsize if label_fontsize is not None else max(4.0, target_inches * 1.4)

    # Labels at centroids, colored to match, with a small stagger.
    placed = []  # (cx, cy) already used, to nudge collisions
    for node in graph.nodes:
        m = masks.get(node.node_id)
        if m is None or m.sum() == 0:
            continue
        ys, xs = np.nonzero(m)
        cx, cy = float(xs.mean()), float(ys.mean())
        # nudge if too close to an existing label
        for (px, py) in placed:
            if abs(px - cx) < W * 0.08 and abs(py - cy) < H * 0.05:
                cy += H * 0.05
        placed.append((cx, cy))

        color = cmap.color(node.node_id)
        tag = "ee" if node.node_type == "ee" else display_name(node.name)
        ax.text(
            cx, cy, tag,
            color="white", fontsize=fs, fontweight="bold",
            ha="center", va="center",
            bbox=dict(facecolor=color, alpha=0.85, pad=0.8,
                      edgecolor="white", linewidth=0.3),
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path
