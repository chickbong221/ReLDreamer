"""Render the semantic graph as a node-link diagram in the reference style:
large filled circles for nodes (ee centred, objects on a ring), bold colored
name below each circle, and plain italic relation labels stacked along each
edge -- no chrome / no chip boxes. Absolute and temporal relations remain
distinguishable by text color (dark vs. muted purple) but share the page.

Layout is deterministic and scales the ring radius + canvas with the number of
objects so dense graphs spread out, with a per-node de-collision nudge so
distinct same-category instances do not stack on top of one another.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schema import Edge, Graph
from .palette import ColorMap


# Absolute-block ordering: spatial first, then physical (planar-distance,
# height-offset, orientation-alignment, then contact / grasp / support). The
# eye lands on "where" before "what kind of contact".
_ABSOLUTE_ORDER = (
    "planar-distance",
    "height-offset",
    "orientation-alignment",
    "contact",
    "grasp",
    "support",
)


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
        # One-object case: a long left -> right horizontal so the edge has
        # room to carry several stacked relation lines.
        pos["ee"] = np.array([-radius * 0.55, 0.0])
        pos[objects[0]] = np.array([radius * 0.55, 0.0])
        return pos
    n = max(len(objects), 1)
    for i, nid in enumerate(objects):
        ang = np.pi / 2 - 2 * np.pi * i / n
        pos[nid] = np.array([radius * np.cos(ang), radius * np.sin(ang)])

    # De-collision: any two object nodes that landed within ``min_sep`` get
    # nudged outward along their own angle until separated. Distinct same-
    # category instances (e.g. two ``024_bowl``s) must remain visible.
    min_sep = 2.0 * node_r + 0.35
    nudge_step = max(node_r * 0.5, 0.18)
    max_iters = 80
    for nid in objects:
        for _ in range(max_iters):
            p = pos[nid]
            r = float(np.linalg.norm(p))
            if r < 1e-9:
                pos[nid] = np.array([0.0, max(min_sep, radius)])
                continue
            unit = p / r
            collides = any(
                other != nid and np.linalg.norm(pos[other] - p) < min_sep
                for other in pos
            )
            if not collides:
                break
            pos[nid] = unit * (r + nudge_step)
    return pos


def _split_labels(elist: List[Edge]) -> Tuple[List[str], List[str]]:
    """Return (absolute_labels, temporal_labels) for an edge group, value-only.

    The label is just the discrete value (``far``, ``contact``,
    ``maintain-grasp`` ...) -- the relation name is omitted because the value
    alone reads cleanly inside the small chip box.
    """
    absolute: List[Edge] = []
    temporal: List[Edge] = []
    for e in elist:
        (temporal if e.temporal else absolute).append(e)

    def _abs_rank(e: Edge) -> Tuple[int, str]:
        try:
            return (_ABSOLUTE_ORDER.index(e.relation), "")
        except ValueError:
            return (len(_ABSOLUTE_ORDER), e.relation)

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
    from matplotlib.patches import Circle

    cmap = colormap or ColorMap()
    cmap.assign_all(graph.node_ids())

    n_obj = sum(1 for n in graph.nodes
                if n.node_type == "object" and n.valid_mask)

    # Big canvas. Even the 1-object panel is rendered large so the labels
    # along the single edge have room to breathe.
    if n_obj <= 1:
        figsize = (11.0, 8.0)
        radius = 3.0
    elif n_obj == 2:
        figsize = (12.0, 9.0)
        radius = 3.2
    else:
        figsize = (
            max(14.0, 10.0 + 0.65 * n_obj),
            max(11.0, 8.0 + 0.55 * n_obj),
        )
        radius = 3.2 + 0.45 * (n_obj - 3)

    node_r = 0.32   # data-unit radius for both the Circle patch and arrow trim

    pos = _radial_layout(graph, radius, node_r)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.axis("off")
    bg = "#fdf0e9"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # ----------------------------------------------------------------- edges
    drawn = [e for e in graph.edges if (not e.masked or not drawable_only)]
    by_pair: Dict[Tuple[str, str], List[Edge]] = {}
    for e in drawn:
        if e.src not in pos or e.dst not in pos:
            continue
        by_pair.setdefault((e.src, e.dst), []).append(e)

    for (src, dst), elist in by_pair.items():
        p0, p1 = pos[src], pos[dst]
        d = p1 - p0
        L = float(np.linalg.norm(d)) + 1e-9
        u = d / L
        a0 = p0 + u * node_r
        a1 = p1 - u * node_r
        is_stale = any(e.stale for e in elist)
        is_support = any(
            (not e.temporal) and e.relation == "support" and not e.masked
            for e in elist
        )

        if is_support:
            edge_color = "#b15a00"
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
            lw = 1.4
            alpha = 0.85
            linestyle = "solid"

        ax.annotate(
            "", xy=a1, xytext=a0,
            arrowprops=dict(
                arrowstyle="-|>", color=edge_color, lw=lw, alpha=alpha,
                linestyle=linestyle,
                shrinkA=0, shrinkB=0,
                mutation_scale=18,
            ),
            zorder=2,
        )

        absolute_labels, temporal_labels = _split_labels(elist)
        if not absolute_labels and not temporal_labels:
            continue

        # Label anchor: midpoint of the trimmed edge, nudged perpendicular to
        # the arrow shaft so the chip box doesn't sit on top of the line.
        mid = a0 + (a1 - a0) * 0.5
        perp = np.array([-u[1], u[0]])
        chip_offset = 0.32

        # Distinct chip backgrounds so absolute and temporal blocks are
        # immediately separable; stale edges override absolute with the blue
        # language used elsewhere for frozen-pose pairs.
        absolute_bg = "#d9ecff" if is_stale else "#fff3c4"   # warm cream
        absolute_edge = "#2f75b5" if is_stale else "#c79a2f"
        absolute_text = "#1c3d6e" if is_stale else "#3a2a05"

        temporal_bg = "#e4dcf5"                              # soft lavender
        temporal_edge = "#6c5aa1"
        temporal_text = "#2c1f5c"

        if absolute_labels and temporal_labels:
            abs_anchor = mid + perp * chip_offset
            tmp_anchor = mid - perp * chip_offset
        elif absolute_labels:
            abs_anchor = mid + perp * (chip_offset * 0.6)
            tmp_anchor = None
        else:
            abs_anchor = None
            tmp_anchor = mid + perp * (chip_offset * 0.6)

        if absolute_labels and abs_anchor is not None:
            ax.text(
                abs_anchor[0], abs_anchor[1], "\n".join(absolute_labels),
                fontsize=9.5, ha="center", va="center", style="italic",
                color=absolute_text, zorder=4, linespacing=1.05,
                bbox=dict(
                    facecolor=absolute_bg, edgecolor=absolute_edge,
                    linewidth=0.7, pad=0.45, alpha=0.96,
                    boxstyle="round,pad=0.35",
                ),
            )
        if temporal_labels and tmp_anchor is not None:
            ax.text(
                tmp_anchor[0], tmp_anchor[1], "\n".join(temporal_labels),
                fontsize=9.5, ha="center", va="center", style="italic",
                color=temporal_text, zorder=4, linespacing=1.05,
                bbox=dict(
                    facecolor=temporal_bg, edgecolor=temporal_edge,
                    linewidth=0.7, pad=0.45, alpha=0.96,
                    boxstyle="round,pad=0.35",
                ),
            )

    # ----------------------------------------------------------------- nodes
    for node in graph.nodes:
        nid = node.node_id
        if nid not in pos or not node.valid_mask:
            continue
        x, y = pos[nid]
        if node.frozen_pose:
            face = (0.29, 0.56, 0.89)
            edge_col = "#1c3d6e"
            linestyle = (0, (3, 2))
        else:
            face = cmap.color(nid)
            edge_col = "#000000" if node.persistent else "white"
            linestyle = "solid"

        alpha = 0.55 if not node.visible else 1.0
        circ = Circle(
            (x, y), node_r,
            facecolor=face, edgecolor=edge_col, linewidth=1.8,
            linestyle=linestyle, alpha=alpha, zorder=3,
        )
        ax.add_patch(circ)

        label = "ee" if node.node_type == "ee" else node.name
        label_color = tuple(0.45 * np.asarray(face))
        ax.text(
            x, y - node_r - 0.18, label,
            fontsize=12.5, fontweight="bold",
            ha="center", va="top",
            color=label_color, zorder=5,
        )

    # ----------------------------------------------------------------- frame
    sub = graph.meta.get("active_subtask")
    title = f"frame {graph.frame}  |  {graph.env_id}"
    if sub:
        title += f"  |  subtask={sub}"
    ax.set_title(title, fontsize=14)

    # Lock view to the actual extents (with padding) so node circles render at
    # their true data-unit size regardless of the axes' auto-scaling.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        pad = node_r + 0.9
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad - 0.4, max(ys) + pad)
    ax.set_aspect("equal")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
