"""Render the semantic graph as a node-link diagram, styled after the draft
figure: colored circular nodes, name label directly under each node, directed
edges with relation labels sitting on the edge midpoint.

Layout is deterministic: ee is placed at center, object nodes are arranged on a
circle around it, so labels never drift and frames are visually stable. Only
drawable (non-masked) edges are shown.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schema import Graph
from .palette import ColorMap


def _radial_layout(graph: Graph) -> Dict[str, np.ndarray]:
    """ee at center; objects evenly spaced on a unit circle around it."""
    pos: Dict[str, np.ndarray] = {}
    objects = [n.node_id for n in graph.nodes if n.node_type == "object"]
    has_ee = graph.get_node("ee") is not None

    if has_ee:
        pos["ee"] = np.array([0.0, 0.0])
    n = max(len(objects), 1)
    # start at top, go clockwise
    for i, nid in enumerate(objects):
        ang = np.pi / 2 - 2 * np.pi * i / n
        r = 1.0 if has_ee else 0.0
        pos[nid] = np.array([r * np.cos(ang), r * np.sin(ang)])
    if not has_ee and objects:
        # no ee: spread objects on a circle centered at origin
        for i, nid in enumerate(objects):
            ang = np.pi / 2 - 2 * np.pi * i / n
            pos[nid] = np.array([np.cos(ang), np.sin(ang)])
    return pos


def _perp_offset(p0: np.ndarray, p1: np.ndarray, k: float) -> np.ndarray:
    """Unit perpendicular to the p0->p1 segment, scaled by k."""
    d = p1 - p0
    norm = np.linalg.norm(d) + 1e-9
    perp = np.array([-d[1], d[0]]) / norm
    return perp * k


def render_graph(
    graph: Graph,
    out_path: str,
    drawable_only: bool = True,
    colormap: Optional[ColorMap] = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = colormap or ColorMap()
    cmap.assign_all(graph.node_ids())

    pos = _radial_layout(graph)
    fig, ax = plt.subplots(figsize=(7, 6.2), dpi=130)
    ax.axis("off")
    bg = "#fdf0e9"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    node_r = 0.12   # in data units; used to shrink arrows to node edge

    # Collect drawable edges, group by unordered pair so we can fan labels.
    drawn = [e for e in graph.edges if (not e.masked or not drawable_only)]
    by_pair: Dict[Tuple[str, str], List] = {}
    for e in drawn:
        if e.src not in pos or e.dst not in pos:
            continue
        by_pair.setdefault((e.src, e.dst), []).append(e)

    # Draw each ordered pair's arrow once; stack its relation labels.
    for (src, dst), elist in by_pair.items():
        p0, p1 = pos[src], pos[dst]
        d = p1 - p0
        L = np.linalg.norm(d) + 1e-9
        u = d / L
        a0 = p0 + u * node_r          # start at node edge
        a1 = p1 - u * node_r          # end at node edge
        ax.annotate(
            "", xy=a1, xytext=a0,
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4),
            zorder=2,
        )
        # label block at midpoint, nudged perpendicular so it clears the line
        mid = (a0 + a1) / 2
        off = _perp_offset(p0, p1, 0.06)
        labels = []
        for e in elist:
            txt = e.label
            labels.append(txt)
        ax.text(
            mid[0] + off[0], mid[1] + off[1], "\n".join(labels),
            fontsize=8, ha="center", va="center", style="italic",
            color="#222", zorder=4,
            bbox=dict(facecolor=bg, edgecolor="none", pad=0.4, alpha=0.9),
        )

    # Nodes + labels on top.
    for node in graph.nodes:
        nid = node.node_id
        if nid not in pos:
            continue
        x, y = pos[nid]
        color = cmap.color(nid)
        size = 1100 if node.node_type == "ee" else 850
        edgecol = "#000" if node.persistent else "white"
        ax.scatter(
            [x], [y], s=size, c=[color], zorder=3,
            edgecolors=edgecol, linewidths=1.6,
            alpha=0.5 if not node.visible else 1.0,
        )
        label = "ee" if node.node_type == "ee" else node.name
        # label directly beneath the node, offset by ~node radius
        ax.text(
            x, y - node_r - 0.06, label,
            fontsize=10, fontweight="bold",
            ha="center", va="top",
            color=tuple(0.55 * np.asarray(color)), zorder=5,
        )

    sub = graph.meta.get("active_subtask")
    title = f"frame {graph.frame}  |  {graph.env_id}"
    if sub:
        title += f"  |  subtask={sub}"
    ax.set_title(title, fontsize=11)
    ax.margins(0.22)
    ax.set_aspect("equal")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
