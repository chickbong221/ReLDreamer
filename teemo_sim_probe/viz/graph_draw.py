"""Render the semantic graph as a node-link diagram, styled after the draft
figure: colored circular nodes, name label directly under each node, directed
edges with relation labels along the edge.

Layout is deterministic: ee at center, object nodes on a circle around it. The
circle radius and figure size scale with the number of objects so dense graphs
(many nodes) stay readable; a post-pass nudges any nodes that landed within a
minimum separation outward along their angle so distinct instances of the same
category do not visually overlap.

Each edge's labels are split into two chips by ``edge.temporal``: absolute
relations on one background colour, temporal ones (``*_change`` / ``*-transition``)
on another, so a glance separates the current frame's state from change history.
Labels are placed toward the object end of each edge so they fan outward instead
of piling up at the centre. Only drawable (non-masked) edges are shown.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schema import Edge, Graph
from .palette import ColorMap


# Absolute physical predicates are listed first within the absolute chip so the
# eye lands on grasp / contact / support before the spatial bin labels.
_PHYSICAL_ABSOLUTE = ("grasp", "support", "contact")


def _radial_layout(
    graph: Graph, radius: float, node_r: float
) -> Dict[str, np.ndarray]:
    pos: Dict[str, np.ndarray] = {}
    objects = [n.node_id for n in graph.nodes
               if n.node_type == "object" and n.valid_mask]
    has_ee = graph.get_node("ee") is not None
    if has_ee:
        pos["ee"] = np.array([0.0, 0.0])
    if has_ee and len(objects) == 1:
        # Compact sparse layout: a short left-to-right semantic statement.
        pos["ee"] = np.array([-0.9, 0.0])
        pos[objects[0]] = np.array([0.9, 0.0])
        return pos
    n = max(len(objects), 1)
    for i, nid in enumerate(objects):
        ang = np.pi / 2 - 2 * np.pi * i / n
        pos[nid] = np.array([radius * np.cos(ang), radius * np.sin(ang)])

    # De-collision: any two object nodes that landed within ``min_sep`` get
    # nudged outward along their own angle until separated. This handles
    # distinct same-category instances (e.g. two ``024_bowl`` instances) that
    # otherwise stack on top of each other -- we keep them as separate nodes
    # rather than merging them.
    min_sep = 2.0 * node_r + 0.18
    nudge_step = max(node_r * 0.5, 0.12)
    max_iters = 60
    for nid in objects:
        for _ in range(max_iters):
            p = pos[nid]
            r = float(np.linalg.norm(p))
            if r < 1e-9:
                # Pathological: send straight up before pushing further out.
                pos[nid] = np.array([0.0, max(min_sep, radius)])
                continue
            unit = p / r
            collides = False
            for other in pos:
                if other == nid:
                    continue
                if np.linalg.norm(pos[other] - p) < min_sep:
                    collides = True
                    break
            if not collides:
                break
            pos[nid] = unit * (r + nudge_step)
    return pos


def _split_labels(
    elist: List[Edge],
) -> Tuple[List[str], List[str]]:
    """Return (absolute_labels, temporal_labels) for an edge group.

    The label is just the discrete value (e.g. ``far``, ``contact``,
    ``maintain-grasp``); the relation name is omitted because the value alone
    reads cleanly inside the small chip box.
    """
    absolute: List[Edge] = []
    temporal: List[Edge] = []
    for e in elist:
        if e.temporal:
            temporal.append(e)
        else:
            absolute.append(e)

    def _abs_rank(e: Edge) -> Tuple[int, str]:
        # Physical predicates lead, then spatial bin labels, alphabetical tie.
        if e.relation in _PHYSICAL_ABSOLUTE:
            return (0, str(_PHYSICAL_ABSOLUTE.index(e.relation)))
        return (1, e.relation)

    absolute.sort(key=_abs_rank)
    temporal.sort(key=lambda e: e.relation)
    return (
        [str(e.label) for e in absolute],
        [str(e.label) for e in temporal],
    )


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

    n_obj = sum(1 for n in graph.nodes
                if n.node_type == "object" and n.valid_mask)
    # Scale radius + canvas with node count so crowded graphs spread out.
    radius = 2.2 + 0.35 * max(n_obj - 3, 0)
    figsize = (6.4, 4.0) if n_obj <= 2 else (
        max(10.0, 7.5 + 0.55 * n_obj), max(8.5, 6.5 + 0.50 * n_obj)
    )

    node_r = 0.22   # data units; arrows shrink to node edge

    pos = _radial_layout(graph, radius, node_r)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.axis("off")
    bg = "#fdf0e9"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # Distinct backgrounds for the two label chips. Absolute uses a light
    # neutral; temporal uses a soft lavender so the change-history block reads
    # as a separate panel without dominating the scene.
    absolute_chip_bg = "#fff8eb"
    temporal_chip_bg = "#ece6f6"
    stale_chip_bg = "#d9ecff"

    drawn = [e for e in graph.edges if (not e.masked or not drawable_only)]
    by_pair: Dict[Tuple[str, str], List[Edge]] = {}
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
        is_stale = any(e.stale for e in elist)
        is_support = any(
            (not e.temporal) and e.relation == "support" and not e.masked
            for e in elist
        )
        if is_support:
            edge_color = "#b15a00"   # warm distinct hue for the support arrow
            lw = 2.2
            alpha = 0.95
            linestyle = "solid"
        elif is_stale:
            edge_color = "#2f75b5"
            lw = 1.5
            alpha = 0.9
            linestyle = (0, (4, 2))
        else:
            edge_color = "#444"
            lw = 1.2
            alpha = 0.8
            linestyle = "solid"
        ax.annotate(
            "", xy=a1, xytext=a0,
            arrowprops=dict(
                arrowstyle="-|>", color=edge_color, lw=lw, alpha=alpha,
                linestyle=linestyle,
            ),
            zorder=2,
        )

        absolute_labels, temporal_labels = _split_labels(elist)
        # Label centre sits 62% along the edge so chips fan outward.
        anchor = a0 + (a1 - a0) * 0.62

        # Stack absolute chip (top) and temporal chip (bottom) so they share
        # the same anchor without overlapping.
        chip_offset = 0.13
        if absolute_labels:
            abs_bg = stale_chip_bg if is_stale else absolute_chip_bg
            ax.text(
                anchor[0], anchor[1] + chip_offset, "\n".join(absolute_labels),
                fontsize=6.5, ha="center", va="center", style="italic",
                color=edge_color if is_stale else "#333", zorder=4,
                linespacing=0.95,
                bbox=dict(
                    facecolor=abs_bg,
                    edgecolor=edge_color if is_stale else "#9b8a73",
                    linewidth=0.6,
                    pad=0.35,
                    alpha=0.95,
                ),
            )
        if temporal_labels:
            ax.text(
                anchor[0], anchor[1] - chip_offset, "\n".join(temporal_labels),
                fontsize=6.5, ha="center", va="center", style="italic",
                color="#3a2c66", zorder=4, linespacing=0.95,
                bbox=dict(
                    facecolor=temporal_chip_bg,
                    edgecolor="#6c5aa1",
                    linewidth=0.6,
                    pad=0.35,
                    alpha=0.95,
                ),
            )

    for node in graph.nodes:
        nid = node.node_id
        if nid not in pos:
            continue
        if not node.valid_mask:
            continue
        x, y = pos[nid]
        # Retained frozen nodes use the same blue language as their stale edges.
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
