"""Render the semantic graph as a node-link diagram in the reference style.

Large filled circles for nodes (ee centred, objects on a ring) with bold
colored labels; plain italic relation values stacked along each edge in
family-colored chips. Relations in the same family share one chip background,
so the viewer reads ``event`` vs ``spatial`` vs ``affordance`` at a glance.

Layout is deterministic and scales the ring radius + canvas with the number
of objects so dense graphs spread out, with a per-node de-collision nudge so
distinct same-category instances do not stack on top of one another.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.schema import Edge, Graph
from .palette import ColorMap


# --------------------------------------------------------------------------- #
# Relation -> family classification
# --------------------------------------------------------------------------- #
_FAMILY_EVENT = "event"
_FAMILY_SPATIAL = "spatial"
_FAMILY_AFFORDANCE = "affordance"

_RELATION_FAMILY: Dict[str, str] = {
    # Event (absolute + transition)
    "contact": _FAMILY_EVENT,
    "grasp": _FAMILY_EVENT,
    "support": _FAMILY_EVENT,
    "contact-transition": _FAMILY_EVENT,
    "grasp-transition": _FAMILY_EVENT,
    "support-transition": _FAMILY_EVENT,
    # Spatial
    "planar-distance": _FAMILY_SPATIAL,
    "height-offset": _FAMILY_SPATIAL,
    "planar-distance-change": _FAMILY_SPATIAL,
    "height-offset-change": _FAMILY_SPATIAL,
    # Affordance compatibility
    "grasp-compatibility": _FAMILY_AFFORDANCE,
    "contact-compatibility": _FAMILY_AFFORDANCE,
    "grasp-compatibility-change": _FAMILY_AFFORDANCE,
    "contact-compatibility-change": _FAMILY_AFFORDANCE,
}

# Family ordering used both for chip-row ordering and (within a chip) the
# absolute-then-temporal stacking.
_FAMILY_ORDER = (_FAMILY_EVENT, _FAMILY_SPATIAL, _FAMILY_AFFORDANCE)

# Within a family, sort absolute relations by this order then temporal ones.
_INTRA_FAMILY_ORDER = {
    _FAMILY_EVENT: ("contact", "grasp", "support",
                    "contact-transition", "grasp-transition", "support-transition"),
    _FAMILY_SPATIAL: ("planar-distance", "height-offset",
                      "planar-distance-change", "height-offset-change"),
    _FAMILY_AFFORDANCE: ("grasp-compatibility", "contact-compatibility",
                         "grasp-compatibility-change",
                         "contact-compatibility-change"),
}

# Per-family chip styling.
_FAMILY_STYLE: Dict[str, Dict[str, str]] = {
    _FAMILY_EVENT:      {"bg": "#ffe0c2", "edge": "#c25a00", "text": "#5a2900"},
    _FAMILY_SPATIAL:    {"bg": "#d4e7ff", "edge": "#2f6ec2", "text": "#13396b"},
    _FAMILY_AFFORDANCE: {"bg": "#e4dcf5", "edge": "#6c5aa1", "text": "#2c1f5c"},
}

# Stale edges override every family chip with the same blue palette used for
# frozen-pose nodes -- a single visual signal that the data is not fresh.
_STALE_STYLE = {"bg": "#d9ecff", "edge": "#2f75b5", "text": "#1c3d6e"}


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
        pos["ee"] = np.array([-radius * 0.55, 0.0])
        pos[objects[0]] = np.array([radius * 0.55, 0.0])
        return pos
    n = max(len(objects), 1)
    for i, nid in enumerate(objects):
        ang = np.pi / 2 - 2 * np.pi * i / n
        pos[nid] = np.array([radius * np.cos(ang), radius * np.sin(ang)])

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


def _family_of(relation: str) -> Optional[str]:
    return _RELATION_FAMILY.get(relation)


def _group_by_family(elist: List[Edge]) -> Dict[str, List[str]]:
    """Bucket an edge group's labels by family in canonical intra-family order.

    Absolute relations come before temporal ones within a family, both follow
    ``_INTRA_FAMILY_ORDER``. Unknown relations are skipped silently.
    """
    grouped: Dict[str, List[Tuple[int, int, str]]] = {f: [] for f in _FAMILY_ORDER}
    for e in elist:
        family = _family_of(e.relation)
        if family is None:
            continue
        order = _INTRA_FAMILY_ORDER.get(family, ())
        try:
            rank = order.index(e.relation)
        except ValueError:
            rank = len(order)
        temporal_rank = 1 if e.temporal else 0
        grouped[family].append((temporal_rank, rank, str(e.label)))

    out: Dict[str, List[str]] = {}
    for family in _FAMILY_ORDER:
        items = sorted(grouped[family])
        if items:
            out[family] = [label for _t, _r, label in items]
    return out


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

    node_r = 0.32

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

        grouped = _group_by_family(elist)
        if not grouped:
            continue

        # Place one chip per family, distributed perpendicular to the edge
        # midpoint. Offsets are symmetric around the line: 1 chip -> 0,
        # 2 chips -> +/-spacing, 3 chips -> -spacing, 0, +spacing.
        mid = a0 + (a1 - a0) * 0.5
        perp = np.array([-u[1], u[0]])
        spacing = 0.42
        families_present = [f for f in _FAMILY_ORDER if f in grouped]
        n = len(families_present)
        if n == 1:
            offsets = [0.0]
        elif n == 2:
            offsets = [-spacing * 0.5, spacing * 0.5]
        else:
            offsets = [-spacing, 0.0, spacing]
        for offset, family in zip(offsets, families_present):
            anchor = mid + perp * offset
            style = _STALE_STYLE if is_stale else _FAMILY_STYLE[family]
            ax.text(
                anchor[0], anchor[1], "\n".join(grouped[family]),
                fontsize=9.5, ha="center", va="center", style="italic",
                color=style["text"], zorder=4, linespacing=1.05,
                bbox=dict(
                    facecolor=style["bg"], edgecolor=style["edge"],
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
