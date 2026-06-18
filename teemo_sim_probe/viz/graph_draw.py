"""Render the semantic graph as a node-link diagram, styled after the draft
figure: colored circular nodes with name labels, directed edges with relation
labels. Only drawable (non-masked) edges are shown.

Uses networkx for layout if available, else a deterministic circular layout.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np

from ..core.schema import Graph


def _color_for(node_id: str) -> Tuple[float, float, float]:
    h = hashlib.md5(node_id.encode()).digest()
    rgb = np.array([h[0], h[1], h[2]], dtype=float) / 255.0
    return tuple(0.35 + 0.55 * rgb)


def _circular_layout(node_ids: List[str]) -> Dict[str, np.ndarray]:
    n = len(node_ids)
    pos = {}
    for i, nid in enumerate(node_ids):
        ang = 2 * np.pi * i / max(n, 1)
        pos[nid] = np.array([np.cos(ang), np.sin(ang)])
    return pos


def _layout(graph: Graph) -> Dict[str, np.ndarray]:
    node_ids = graph.node_ids()
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        for e in graph.edges:
            if not e.masked:
                G.add_edge(e.src, e.dst)
        # ee in the centre tends to read well; seed spring layout from circle.
        pos0 = _circular_layout(node_ids)
        pos = nx.spring_layout(G, pos=pos0, k=1.2, seed=0)
        return {k: np.asarray(v) for k, v in pos.items()}
    except Exception:
        return _circular_layout(node_ids)


def render_graph(graph: Graph, out_path: str, drawable_only: bool = True) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = _layout(graph)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.axis("off")
    ax.set_facecolor("#fdf0e9")
    fig.patch.set_facecolor("#fdf0e9")

    # Edges first.
    drawn = [e for e in graph.edges if (e.masked is False or not drawable_only)]
    # Collapse multiple relations on same ordered pair onto stacked labels.
    pair_labels: Dict[Tuple[str, str], List[str]] = {}
    for e in drawn:
        pair_labels.setdefault((e.src, e.dst), []).append(
            e.label if not e.temporal else f"{e.label}"
        )

    for (src, dst), labels in pair_labels.items():
        if src not in pos or dst not in pos:
            continue
        p0, p1 = pos[src], pos[dst]
        ax.annotate(
            "", xy=p1, xytext=p0,
            arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.4,
                            shrinkA=18, shrinkB=18),
        )
        mid = (p0 + p1) / 2
        ax.text(
            mid[0], mid[1], "\n".join(labels),
            fontsize=8, ha="center", va="center", style="italic",
            color="#222",
            bbox=dict(facecolor="#fdf0e9", edgecolor="none", pad=0.5),
        )

    # Nodes on top.
    for nid in graph.node_ids():
        if nid not in pos:
            continue
        node = graph.get_node(nid)
        x, y = pos[nid]
        color = _color_for(nid)
        size = 900 if node.node_type == "ee" else 700
        edgecol = "#000" if node.persistent else "none"
        ax.scatter([x], [y], s=size, c=[color], zorder=3,
                   edgecolors=edgecol, linewidths=1.5,
                   alpha=0.5 if not node.visible else 1.0)
        label = "ee" if node.node_type == "ee" else node.name
        ax.text(x, y - 0.16, label, fontsize=10, fontweight="bold",
                ha="center", va="top", color=tuple(0.6 * np.asarray(color)))

    ax.set_title(
        f"frame {graph.frame}  |  {graph.env_id}"
        + (f"  |  subtask={graph.meta.get('active_subtask')}"
           if graph.meta.get("active_subtask") else ""),
        fontsize=10,
    )
    ax.margins(0.18)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
