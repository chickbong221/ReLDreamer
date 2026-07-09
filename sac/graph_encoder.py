"""sg2im-style graph encoder.

Padded backing tensors (B, N_max) / (B, E_max) plus n_nodes/n_edges counts are
converted to a flat sg2im-style batch (obj_vecs, pred_vecs, edges, node_to_sample)
inside forward. Categorical predicates only. L layers of GraphTripleConv on
the flat batch, followed by a parametric attention readout with a learned
pool query and Q/K/V projections over [node_conf ; obj_vec] inputs.

Target conditioning: nodes matching ``graph_target_id`` get a learned marker
added before message passing, and the pool query is shifted by the target's
node embedding (zero when the target id is pad).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn


def _mlp(dims: List[int], last_act: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if last_act or i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class GraphTripleConv(nn.Module):
    """Flat sg2im GraphTripleConv: (O, D), (T, D), (T, 2) -> (O, D), (T, D)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net1 = _mlp(
            [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim], last_act=False,
        )
        self.net2 = _mlp([hidden_dim, hidden_dim, output_dim], last_act=False)

    def forward(
        self,
        obj_vecs: torch.Tensor,   # (O, D_in)
        pred_vecs: torch.Tensor,  # (T, D_in)
        edges: torch.Tensor,      # (T, 2) long
    ):
        O = obj_vecs.size(0)
        T = pred_vecs.size(0)
        H, D_out = self.hidden_dim, self.output_dim
        dtype, device = obj_vecs.dtype, obj_vecs.device

        s_idx = edges[:, 0]
        o_idx = edges[:, 1]

        cur_s = obj_vecs[s_idx]
        cur_o = obj_vecs[o_idx]
        new_t = self.net1(torch.cat([cur_s, pred_vecs, cur_o], dim=-1))

        new_s = new_t[:, :H]
        new_p = new_t[:, H:H + D_out]
        new_o = new_t[:, H + D_out:2 * H + D_out]

        pooled = torch.zeros(O, H, dtype=dtype, device=device)
        pooled.scatter_add_(0, s_idx.unsqueeze(-1).expand(-1, H), new_s)
        pooled.scatter_add_(0, o_idx.unsqueeze(-1).expand(-1, H), new_o)

        counts = torch.zeros(O, dtype=dtype, device=device)
        ones = torch.ones(T, dtype=dtype, device=device)
        counts.scatter_add_(0, s_idx, ones)
        counts.scatter_add_(0, o_idx, ones)
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)

        return self.net2(pooled), new_p


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 512,
        out_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.node_emb = nn.Embedding(node_vocab_size, embed_dim, padding_idx=0)
        self.edge_emb = nn.Embedding(edge_vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            GraphTripleConv(embed_dim, embed_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        self.pool_query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.target_marker = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.W_K = nn.Linear(embed_dim + 1, embed_dim)
        self.W_V = nn.Linear(embed_dim + 1, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)
        for m in (self.W_K, self.W_V, self.W_O):
            m.apply(_orthogonal_init)

        self.attn_scale = 1.0 / math.sqrt(embed_dim)
        self.readout = _mlp([embed_dim, out_dim, out_dim], last_act=False)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_ids = obs["graph_node_ids"].long()
        node_conf = obs["graph_node_conf"].float()
        edge_src = obs["graph_edge_src"].long()
        edge_dst = obs["graph_edge_dst"].long()
        edge_pred = obs["graph_edge_pred"].long()
        n_nodes = obs["graph_n_nodes"].long()
        n_edges = obs["graph_n_edges"].long()
        target_id = obs["graph_target_id"].long()

        B, N_max = node_ids.shape
        E_max = edge_src.size(1)
        device = node_ids.device
        D = self.embed_dim

        node_arange = torch.arange(N_max, device=device).unsqueeze(0)
        node_mask = node_arange < n_nodes.unsqueeze(1)              # (B, N_max)
        node_ids_flat = node_ids[node_mask]                          # (O,)
        node_conf_flat = node_conf[node_mask]                        # (O,)

        O_total = int(node_ids_flat.size(0))
        if O_total == 0:
            return torch.zeros(B, self.out_dim, device=device, dtype=node_conf.dtype)

        sample_grid_n = torch.arange(B, device=device).unsqueeze(1).expand(-1, N_max)
        node_to_sample = sample_grid_n[node_mask]                    # (O,)

        node_offset = torch.zeros(B, device=device, dtype=torch.long)
        node_offset[1:] = n_nodes[:-1].cumsum(0)

        edge_arange = torch.arange(E_max, device=device).unsqueeze(0)
        edge_mask = edge_arange < n_edges.unsqueeze(1)               # (B, E_max)
        edge_src_local = edge_src[edge_mask]
        edge_dst_local = edge_dst[edge_mask]
        edge_pred_flat = edge_pred[edge_mask]

        sample_grid_e = torch.arange(B, device=device).unsqueeze(1).expand(-1, E_max)
        edge_to_sample = sample_grid_e[edge_mask]
        per_edge_offset = node_offset[edge_to_sample]
        edge_src_flat = edge_src_local + per_edge_offset
        edge_dst_flat = edge_dst_local + per_edge_offset

        obj_vecs = self.node_emb(node_ids_flat)                      # (O, D)
        # Valid node ids are >= 1, so a pad target (0) marks nothing.
        is_target = (node_ids_flat == target_id[node_to_sample]).to(obj_vecs.dtype)
        obj_vecs = obj_vecs + is_target.unsqueeze(-1) * self.target_marker
        pred_vecs = self.edge_emb(edge_pred_flat)                    # (T, D)
        edges = torch.stack([edge_src_flat, edge_dst_flat], dim=-1)  # (T, 2)

        for conv in self.convs:
            obj_vecs, pred_vecs = conv(obj_vecs, pred_vecs, edges)

        x_in = torch.cat([node_conf_flat.unsqueeze(-1), obj_vecs], dim=-1)  # (O, D+1)
        K = self.W_K(x_in)                                           # (O, D)
        V = self.W_V(x_in)                                           # (O, D)

        # Per-sample query: pad target embeds to zero (padding_idx), leaving
        # the plain learned query.
        q = self.pool_query.unsqueeze(0) + self.node_emb(target_id)  # (B, D)
        scores = (K * q[node_to_sample]).sum(-1) * self.attn_scale

        sample_max = torch.full((B,), float("-inf"), device=device, dtype=scores.dtype)
        sample_max = sample_max.scatter_reduce(
            0, node_to_sample, scores.detach(), reduce="amax", include_self=True,
        )
        sample_max = torch.where(
            torch.isinf(sample_max), torch.zeros_like(sample_max), sample_max,
        )
        exp_scores = (scores - sample_max[node_to_sample]).exp()
        denom = torch.zeros(B, device=device, dtype=exp_scores.dtype)
        denom.scatter_add_(0, node_to_sample, exp_scores)
        attn = exp_scores / denom[node_to_sample].clamp(min=1e-8)    # (O,)

        weighted = V * attn.unsqueeze(-1)
        pooled = torch.zeros(B, D, device=device, dtype=V.dtype)
        pooled.scatter_add_(0, node_to_sample.unsqueeze(-1).expand(-1, D), weighted)

        return self.readout(self.W_O(pooled))


GRAPH_OBS_KEYS = (
    "graph_node_ids", "graph_node_ee_mask", "graph_node_conf",
    "graph_edge_src", "graph_edge_dst", "graph_edge_pred",
    "graph_n_nodes", "graph_n_edges", "graph_target_id",
)


def has_graph_obs(sample_obs: Dict[str, torch.Tensor]) -> bool:
    return all(k in sample_obs for k in GRAPH_OBS_KEYS)
