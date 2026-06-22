"""Render the semantic graph as a node-link diagram, styled after the draft
figure: colored circular nodes, name label directly under each node, directed
edges with relation labels along the edge.

Layout is deterministic: ee at center, object nodes on a circle around it. The
circle radius and figure size scale with the number of objects so dense graphs
(many nodes) stay readable. Edge labels are placed toward the object end of each
edge (not the midpoint) so they fan outward instead of piling up at the center.
Only drawable (non-masked) edges are shown.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schema import Graph
from .palette import ColorMap


def _radial_layout(graph: Graph, radius: float) -> Dict[str, np.ndarray]:
    pos: Dict[str, np.ndarray] = {}
    objects = [n.node_id for n in graph.nodes if n.node_type == "object"]
    has_ee = graph.get_node("ee") is not None
    if has_ee:
        pos["ee"] = np.array([0.0, 0.0])
    n = max(len(objects), 1)
    r = radius if has_ee else 0.0
    for i, nid in enumerate(objects):
        ang = np.pi / 2 - 2 * np.pi * i / n
        rr = radius if not has_ee else r
        pos[nid] = np.array([rr * np.cos(ang), rr * np.sin(ang)])
    return pos


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

    n_obj = sum(1 for n in graph.nodes if n.node_type == "object")
    # Scale radius + canvas with node count so crowded graphs spread out.
    radius = 2.2 + 0.18 * max(n_obj - 4, 0)
    figsize = (max(9, 7 + 0.45 * n_obj), max(8, 6 + 0.40 * n_obj))

    pos = _radial_layout(graph, radius)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.axis("off")
    bg = "#fdf0e9"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    node_r = 0.22   # data units; arrows shrink to node edge

    drawn = [e for e in graph.edges if (not e.masked or not drawable_only)]
    by_pair: Dict[Tuple[str, str], List] = {}
    for e in drawn:
        if e.src not in pos or e.dst not in pos:
            continue
        by_pair.setdefault((e.src, e.dst), []).append(e)

    for (src, dst), elist in by_pair.items():
        p0, p1 = pos[src], pos[dst]
        d = p1 - p0
        L = np.linalg.norm(d) + 1e-9
        u = d / L
        a0 = p0 + u * node_r
        a1 = p1 - u * node_r
        ax.annotate(
            "", xy=a1, xytext=a0,
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2,
                            alpha=0.8),
            zorder=2,
        )
        # Label toward the object end (65% along) so labels fan outward and
        # don't stack at the center where all edges converge.
        anchor = a0 + (a1 - a0) * 0.62
        labels = [e.label for e in elist]
        ax.text(
            anchor[0], anchor[1], "\n".join(labels),
            fontsize=6.5, ha="center", va="center", style="italic",
            color="#333", zorder=4, linespacing=0.95,
            bbox=dict(facecolor=bg, edgecolor="none", pad=0.2, alpha=0.85),
        )

    for node in graph.nodes:
        nid = node.node_id
        if nid not in pos:
            continue
        x, y = pos[nid]
        # Retained-with-frozen-pose nodes get a distinctive blue fill + dashed
        # outline, so they read differently from MS-HAB active-target persistents
        # (which still receive fresh poses from SAPIEN).
        if node.frozen_pose:
            color = (0.29, 0.56, 0.89)        # #4a90e2
            edgecol = "#1c3d6e"
            linestyle = (0, (3, 2))
        else:
            color = cmap.color(nid)
            edgecol = "#000" if node.persistent else "white"
            linestyle = "solid"
        size = 1000 if node.node_type == "ee" else 780
        ax.scatter(
            [x], [y], s=size, c=[color], zorder=3,
            edgecolors=edgecol, linewidths=1.5,
            linestyle=linestyle,
            alpha=0.5 if not node.visible else 1.0,
        )
        label = "ee" if node.node_type == "ee" else node.name
        ax.text(
            x, y - node_r - 0.10, label,
            fontsize=8.5, fontweight="bold",
            ha="center", va="top",
            color=tuple(0.55 * np.asarray(color)), zorder=5,
        )

    sub = graph.meta.get("active_subtask")
    title = f"frame {graph.frame}  |  {graph.env_id}"
    if sub:
        title += f"  |  subtask={sub}"
    ax.set_title(title, fontsize=12)
    ax.margins(0.18)
    ax.set_aspect("equal")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
