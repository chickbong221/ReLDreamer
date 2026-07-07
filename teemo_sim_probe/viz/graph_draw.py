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

from ..core.entity_identity import display_name
from ..core.schema import Edge, Graph
from .palette import ColorMap


# --------------------------------------------------------------------------- #
# Relation -> family classification
# --------------------------------------------------------------------------- #
# Three TEEMO families. Physical-state replaces the old ``event`` family;
# transition edges have been removed from the vocabulary entirely.
_FAMILY_PHYSICAL_STATE = "physical_state"
_FAMILY_SPATIAL = "spatial"
_FAMILY_AFFORDANCE = "affordance"

_RELATION_FAMILY: Dict[str, str] = {
    # Physical state (absolute only)
    "contact": _FAMILY_PHYSICAL_STATE,
    "grasp": _FAMILY_PHYSICAL_STATE,
    "support": _FAMILY_PHYSICAL_STATE,
    "contain": _FAMILY_PHYSICAL_STATE,
    # Spatial
    "planar-distance": _FAMILY_SPATIAL,
    "height-offset": _FAMILY_SPATIAL,
    "planar-distance-change": _FAMILY_SPATIAL,
    "height-offset-change": _FAMILY_SPATIAL,
    # Affordance compatibility
    "grasp-compatibility": _FAMILY_AFFORDANCE,
    "contact-compatibility": _FAMILY_AFFORDANCE,
    "support-compatibility": _FAMILY_AFFORDANCE,
    "contain-compatibility": _FAMILY_AFFORDANCE,
    "grasp-compatibility-change": _FAMILY_AFFORDANCE,
    "contact-compatibility-change": _FAMILY_AFFORDANCE,
    "support-compatibility-change": _FAMILY_AFFORDANCE,
    "contain-compatibility-change": _FAMILY_AFFORDANCE,
}

# Family ordering used both for chip-row ordering and (within a chip) the
# absolute-then-temporal stacking.
_FAMILY_ORDER = (_FAMILY_PHYSICAL_STATE, _FAMILY_SPATIAL, _FAMILY_AFFORDANCE)

# Within a family, sort absolute relations by this order then temporal ones.
_INTRA_FAMILY_ORDER = {
    _FAMILY_PHYSICAL_STATE: ("contact", "grasp", "support", "contain"),
    _FAMILY_SPATIAL: ("planar-distance", "height-offset",
                      "planar-distance-change", "height-offset-change"),
    _FAMILY_AFFORDANCE: ("grasp-compatibility", "contact-compatibility",
                         "support-compatibility", "contain-compatibility",
                         "grasp-compatibility-change",
                         "contact-compatibility-change",
                         "support-compatibility-change",
                         "contain-compatibility-change"),
}

# Per-family chip styling. Affordance uses a green palette so it doesn't
# read as the same family as the (also light+cool) spatial blue.
_FAMILY_STYLE: Dict[str, Dict[str, str]] = {
    _FAMILY_PHYSICAL_STATE: {"bg": "#ffe0c2", "edge": "#c25a00", "text": "#5a2900"},
    _FAMILY_SPATIAL:        {"bg": "#d4e7ff", "edge": "#2f6ec2", "text": "#13396b"},
    _FAMILY_AFFORDANCE:     {"bg": "#d8f0dc", "edge": "#3a8f5b", "text": "#1c4a2b"},
}

# Minimum axis-unit separation between chip centers before we consider two
# chips as "overlapping" and nudge the later one along its edge's perp.
# Tuned against the multi-line spatial chips (up to four rows), which are
# the largest labels in the graph. Scaled with the current relation fontsize
# (13pt) so a taller chip still finds space.
_CHIP_MIN_SEP_X = 1.2
_CHIP_MIN_SEP_Y = 0.65
_CHIP_NUDGE_STEP = 0.5
_CHIP_NUDGE_MAX_ITERS = 12

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

    # Square canvas that matches the overlay panels (6" @ 200 dpi -> 1200 px)
    # so the three panels hstack without any padding in the video. The viewport
    # is sized just above the largest expected ring + a chip's worth of margin
    # to keep the graph filling the panel rather than sitting in the middle.
    figsize = (6.0, 6.0)
    if n_obj <= 1:
        radius = 3.0
    elif n_obj == 2:
        radius = 3.2
    else:
        radius = min(3.2 + 0.45 * (n_obj - 3), 5.0)

    node_r = 0.32
    view_half = 6.0

    pos = _radial_layout(graph, radius, node_r)
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
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

    # Track every placed chip center across ALL edges so a chip laid down
    # for a later edge can nudge itself away from earlier chips.
    placed_chip_centers: List[Tuple[float, float]] = []

    for (src, dst), elist in by_pair.items():
        p0, p1 = pos[src], pos[dst]
        d = p1 - p0
        L = float(np.linalg.norm(d)) + 1e-9
        u = d / L
        a0 = p0 + u * node_r
        a1 = p1 - u * node_r
        is_stale = any(e.stale for e in elist)
        is_directed_physical = any(
            (not e.temporal) and e.relation in ("support", "contain")
            and not e.masked
            for e in elist
        )

        if is_directed_physical:
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

        # Chips stack SCREEN-VERTICALLY at the edge midpoint, ordered
        # top-to-bottom as physical / spatial / affordance -- regardless
        # of edge angle. Cross-edge collisions are resolved by nudging
        # the later chip further up or down, so the stack stays vertical
        # instead of drifting off to the side.
        mid = a0 + (a1 - a0) * 0.5
        spacing = 0.55
        families_present = [f for f in _FAMILY_ORDER if f in grouped]
        n = len(families_present)
        if n == 1:
            y_offsets = [0.0]
        elif n == 2:
            y_offsets = [spacing * 0.5, -spacing * 0.5]
        else:
            y_offsets = [spacing, 0.0, -spacing]

        for y_offset, family in zip(y_offsets, families_present):
            anchor = np.array([float(mid[0]), float(mid[1] + y_offset)])
            # Direction the chip prefers to escape in when de-colliding:
            # top-slot pushes up, bottom-slot pushes down. The centre slot
            # (spatial when three families present) picks its direction
            # from the first collision it hits so it doesn't oscillate.
            push_sign = 1.0 if y_offset > 0 else (-1.0 if y_offset < 0 else 0.0)
            for _ in range(_CHIP_NUDGE_MAX_ITERS):
                collided = False
                first_hit_side = 0.0
                for px, py in placed_chip_centers:
                    if (
                        abs(anchor[0] - px) < _CHIP_MIN_SEP_X
                        and abs(anchor[1] - py) < _CHIP_MIN_SEP_Y
                    ):
                        collided = True
                        first_hit_side = 1.0 if anchor[1] >= py else -1.0
                        break
                if not collided:
                    break
                step_sign = push_sign if push_sign != 0.0 else first_hit_side
                if step_sign == 0.0:
                    step_sign = 1.0
                anchor = anchor + np.array([0.0, _CHIP_NUDGE_STEP * step_sign])
            placed_chip_centers.append((float(anchor[0]), float(anchor[1])))
            style = _STALE_STYLE if is_stale else _FAMILY_STYLE[family]
            ax.text(
                anchor[0], anchor[1], "\n".join(grouped[family]),
                fontsize=13.0, ha="center", va="center", style="italic",
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

        label = "ee" if node.node_type == "ee" else display_name(node.name)
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

    ax.set_xlim(-view_half, view_half)
    ax.set_ylim(-view_half, view_half)
    ax.set_aspect("equal")
    # No ``bbox_inches='tight'``: cropping to visible content would defeat the
    # fixed figsize/viewport and re-introduce per-frame size drift.
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
