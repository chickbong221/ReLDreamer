"""Graph encoder for SAC.

Structure per Q1-Q7 design:
  node token    = Emb(c_i)                           -- category only
  edge token    = Emb(r_k)                           -- flat (relation:label)
  L layers of MaskedGraphTripleConv (sg2im-style)
  staleness    = alpha_i = exp(-tau_i / K_soft), appended AFTER the last conv
  readout      = MLP([x_ee ; MeanPool(X_obj)])

Padding is masked out of the graph conv (scatter_add uses edge_valid so invalid
triples contribute zero to pooled features and to the divisor) and out of the
readout mean.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _mlp(dims: List[int], last_act: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if last_act or i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class MaskedGraphTripleConv(nn.Module):
    """sg2im GraphTripleConv with an edge_valid mask.

    Invalid triples multiply their post-net1 subject/predicate/object outputs
    by 0 before scatter_add, and contribute 0 to the pooling divisor. Padded
    nodes with no incoming valid triples end up pooled = 0.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net1 = _mlp(
            [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim],
            last_act=False,
        )
        self.net2 = _mlp([hidden_dim, hidden_dim, output_dim], last_act=False)

    def forward(
        self,
        obj_vecs: torch.Tensor,     # (B, N, D_in)
        pred_vecs: torch.Tensor,    # (B, T, D_in)
        edges: torch.Tensor,        # (B, T, 2) long -- [s_idx, o_idx]
        edge_valid: torch.Tensor,   # (B, T) float
    ):
        B, N, D_in = obj_vecs.shape
        T = pred_vecs.size(1)
        H, D_out = self.hidden_dim, self.output_dim

        s_idx = edges[..., 0]                                  # (B, T)
        o_idx = edges[..., 1]                                  # (B, T)
        s_gather = s_idx.unsqueeze(-1).expand(-1, -1, D_in)
        o_gather = o_idx.unsqueeze(-1).expand(-1, -1, D_in)
        cur_s = torch.gather(obj_vecs, 1, s_gather)            # (B, T, D_in)
        cur_o = torch.gather(obj_vecs, 1, o_gather)            # (B, T, D_in)

        t_vecs = torch.cat([cur_s, pred_vecs, cur_o], dim=-1)  # (B, T, 3*D_in)
        new_t = self.net1(t_vecs)                              # (B, T, 2H + D_out)

        m = edge_valid.unsqueeze(-1)
        new_s = new_t[..., :H] * m
        new_p = new_t[..., H:H + D_out] * m
        new_o = new_t[..., H + D_out:2 * H + D_out] * m

        pooled = torch.zeros(B, N, H, device=obj_vecs.device, dtype=obj_vecs.dtype)
        pooled.scatter_add_(1, s_idx.unsqueeze(-1).expand(-1, -1, H), new_s)
        pooled.scatter_add_(1, o_idx.unsqueeze(-1).expand(-1, -1, H), new_o)

        counts = torch.zeros(B, N, device=obj_vecs.device, dtype=obj_vecs.dtype)
        counts.scatter_add_(1, s_idx, edge_valid)
        counts.scatter_add_(1, o_idx, edge_valid)
        counts = counts.clamp(min=1.0).unsqueeze(-1)
        pooled = pooled / counts

        new_obj = self.net2(pooled)                            # (B, N, D_out)
        return new_obj, new_p


class GraphEncoder(nn.Module):
    """Two-layer masked triple conv + [x_ee ; MeanPool(X_obj)] readout."""

    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.out_dim = out_dim
        # padding_idx=0 forces the pad row to a zero embedding with no grad.
        self.node_emb = nn.Embedding(node_vocab_size, embed_dim, padding_idx=0)
        self.edge_emb = nn.Embedding(edge_vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            MaskedGraphTripleConv(embed_dim, embed_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        readout_in = 2 * (embed_dim + 1)
        self.readout = _mlp([readout_in, out_dim, out_dim], last_act=False)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_ids = obs["graph_node_ids"].long()
        node_valid = obs["graph_node_valid"].float()
        ee_mask = obs["graph_node_ee_mask"].float()
        node_conf = obs["graph_node_conf"].float()
        edge_src = obs["graph_edge_src"].long()
        edge_dst = obs["graph_edge_dst"].long()
        edge_pred = obs["graph_edge_pred"].long()
        edge_valid = obs["graph_edge_valid"].float()

        obj_vecs = self.node_emb(node_ids)                  # (B, N, D)
        pred_vecs = self.edge_emb(edge_pred)                # (B, T, D)
        edges = torch.stack([edge_src, edge_dst], dim=-1)   # (B, T, 2)

        for conv in self.convs:
            obj_vecs, pred_vecs = conv(obj_vecs, pred_vecs, edges, edge_valid)

        obj_vecs = torch.cat([obj_vecs, node_conf.unsqueeze(-1)], dim=-1)  # (B, N, D+1)

        ee_denom = ee_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        x_ee = (obj_vecs * ee_mask.unsqueeze(-1)).sum(dim=1) / ee_denom

        obj_mask = node_valid * (1.0 - ee_mask)
        obj_denom = obj_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        x_obj = (obj_vecs * obj_mask.unsqueeze(-1)).sum(dim=1) / obj_denom

        return self.readout(torch.cat([x_ee, x_obj], dim=-1))


GRAPH_OBS_KEYS = (
    "graph_node_ids", "graph_node_valid", "graph_node_ee_mask", "graph_node_conf",
    "graph_edge_src", "graph_edge_dst", "graph_edge_pred", "graph_edge_valid",
)


def has_graph_obs(sample_obs: Dict[str, torch.Tensor]) -> bool:
    return all(k in sample_obs for k in GRAPH_OBS_KEYS)
