"""SAC modules ported from mshab/agents/sac. Encoder gains an optional graph
branch: gradients from actor loss detach through it and the target encoder
EMA covers it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return int(np.prod(layers(x).shape))


def gaussian_logprob(noise, log_std):
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh(),
        )
        self.out_dim = out_dim
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SharedCNN(nn.Module):
    def __init__(self, pixel_obs_shape, features, filters, strides, padding):
        super().__init__()
        assert len(pixel_obs_shape) == 3
        in_features = [pixel_obs_shape[0]] + list(features)
        layers = []
        for i, (in_f, out_f, k, s) in enumerate(
            zip(in_features, features, filters, strides)
        ):
            layers.append(nn.Conv2d(in_f, out_f, k, s, padding=padding))
            if i < len(filters) - 1:
                layers.append(nn.ReLU())
        layers.append(Flatten())
        self.layers = nn.Sequential(*layers)
        self.out_dim = get_out_shape(pixel_obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, pixels: torch.Tensor):
        # Fold frame-stacked [B, fs, C, H, W] into channels.
        if pixels.dim() == 5:
            b, fs, d, h, w = pixels.shape
            pixels = pixels.view(b, fs * d, h, w).contiguous()
        return self.layers(pixels)


class Encoder(nn.Module):
    """Per-camera CNNs + state projection, with an optional graph branch."""

    def __init__(
        self,
        cnns: nn.ModuleDict,
        pixels_projections: nn.ModuleDict,
        state_projection: RLProjection,
        graph_encoder: Optional[nn.Module] = None,
        graph_projection: Optional[RLProjection] = None,
        graph_actor_gradient: bool = False,
    ):
        super().__init__()
        self.cnns = cnns
        self.pixels_projections = pixels_projections
        self.state_projection = state_projection
        self.graph_encoder = graph_encoder
        self.graph_projection = graph_projection
        self.graph_actor_gradient = bool(graph_actor_gradient)
        self.out_dim = (
            sum(p.out_dim for p in pixels_projections.values())
            + state_projection.out_dim
            + (graph_projection.out_dim if graph_projection is not None else 0)
        )

    def forward(
        self,
        pixels: Dict[str, torch.Tensor],
        state: torch.Tensor,
        graph: Optional[Dict[str, torch.Tensor]] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        pencs = [(k, cnn(pixels[k])) for k, cnn in self.cnns.items()]
        if detach:
            pencs = [(k, p.detach()) for k, p in pencs]
        parts = [self.pixels_projections[k](p) for k, p in pencs]
        parts.append(self.state_projection(state))
        if self.graph_encoder is not None:
            g = self.graph_encoder(graph)
            if detach and not self.graph_actor_gradient:
                g = g.detach()
            parts.append(self.graph_projection(g))
        return torch.cat(parts, dim=1)


class Actor(nn.Module):
    def __init__(self, encoder, action_dim, hidden_dims, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        in_dims = [encoder.out_dim] + list(hidden_dims)
        out_dims = list(hidden_dims) + [2 * action_dim]
        layers = []
        for i, (i_, o_) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(i_, o_))
            if i < len(in_dims) - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.apply(weight_init)

    def forward(
        self,
        pixels: Dict[str, torch.Tensor],
        state: torch.Tensor,
        graph: Optional[Dict[str, torch.Tensor]] = None,
        compute_pi: bool = True,
        compute_log_pi: bool = True,
        detach: bool = False,
    ):
        x = self.encoder(pixels, state, graph, detach=detach)
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        if compute_pi:
            noise = torch.randn_like(mu)
            pi = mu + noise * log_std.exp()
        else:
            pi, noise = None, None

        log_pi = gaussian_logprob(noise, log_std) if compute_log_pi and noise is not None else None
        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, layer_norm=False, dropout=None):
        super().__init__()
        in_dims = [obs_dim + action_dim] + list(hidden_dims)
        out_dims = list(hidden_dims) + [1]
        layers = []
        for i, (i_, o_) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(i_, o_))
            if i < len(in_dims) - 1:
                if dropout is not None and dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                if layer_norm:
                    layers.append(nn.LayerNorm(o_))
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs, action):
        return self.mlp(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
    def __init__(self, encoder, action_dim, hidden_dims, layer_norm=False, dropout=None):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(encoder.out_dim, action_dim, hidden_dims, layer_norm, dropout)
        self.Q2 = QFunction(encoder.out_dim, action_dim, hidden_dims, layer_norm, dropout)
        self.apply(weight_init)

    def forward(
        self,
        pixels: Dict[str, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
        graph: Optional[Dict[str, torch.Tensor]] = None,
        detach: bool = False,
    ):
        x = self.encoder(pixels, state, graph, detach=detach)
        return self.Q1(x, action), self.Q2(x, action)
