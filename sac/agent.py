"""SAC agent: actor, twin critics + target, optimizers, learned alpha.

Structure mirrors mshab/agents/sac. CNNs (and the graph encoder, if enabled)
are shared between actor and critic; the target critic has its own copies
that only move via EMA. Actor loss forwards with detach=True, so shared
encoder parameters only get gradients from the critic loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import GraphEncoder
from .nets import Actor, Critic, Encoder, RLProjection, SharedCNN


@dataclass
class UpdateMetrics:
    critic_loss: float
    q1: float
    q2: float
    actor_loss: Optional[float]
    alpha: float
    alpha_loss: Optional[float]
    entropy_mean: Optional[float]
    log_alpha: float
    target_entropy: float


def _cnn_dict(pixels_obs_space: spaces.Dict, cfg: dict) -> nn.ModuleDict:
    return nn.ModuleDict({
        k: SharedCNN(
            pixel_obs_shape=space.shape,
            features=cfg["cnn_features"],
            filters=cfg["cnn_filters"],
            strides=cfg["cnn_strides"],
            padding=cfg["cnn_padding"],
        )
        for k, space in pixels_obs_space.items()
    })


def _encoder(cnns: nn.ModuleDict, state_dim: int, cfg: dict,
              graph_enc: Optional[GraphEncoder]) -> Encoder:
    pixels_projections = nn.ModuleDict({
        k: RLProjection(cnn.out_dim, cfg["encoder_pixels_feature_dim"])
        for k, cnn in cnns.items()
    })
    state_projection = RLProjection(state_dim, cfg["encoder_state_feature_dim"])
    graph_projection = (
        RLProjection(graph_enc.out_dim, graph_enc.out_dim)
        if graph_enc is not None else None
    )
    return Encoder(cnns, pixels_projections, state_projection,
                    graph_enc, graph_projection)


class SACAgent(nn.Module):
    def __init__(
        self,
        pixels_obs_space: spaces.Dict,
        state_dim: int,
        action_shape,
        cfg: dict,
        graph_cfg: Optional[dict],
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.gamma = float(cfg["gamma"])
        self.critic_tau = float(cfg["critic_tau"])
        self.critic_encoder_tau = float(cfg["critic_encoder_tau"])
        self.actor_update_freq = int(cfg["actor_update_freq"])
        self.critic_target_update_freq = int(cfg["critic_target_update_freq"])

        action_dim = int(np.prod(action_shape))

        shared_cnns = _cnn_dict(pixels_obs_space, cfg)
        target_cnns = _cnn_dict(pixels_obs_space, cfg)

        if graph_cfg is not None:
            def _mk_graph():
                return GraphEncoder(
                    node_vocab_size=graph_cfg["node_vocab_size"],
                    edge_vocab_size=graph_cfg["edge_vocab_size"],
                    embed_dim=graph_cfg["embed_dim"],
                    hidden_dim=graph_cfg["hidden_dim"],
                    out_dim=graph_cfg["out_dim"],
                    num_layers=graph_cfg["num_layers"],
                )
            shared_graph = _mk_graph()
            target_graph = _mk_graph()
        else:
            shared_graph = target_graph = None

        self.actor = Actor(
            _encoder(shared_cnns, state_dim, cfg, shared_graph),
            action_dim=action_dim,
            hidden_dims=cfg["actor_hidden_dims"],
            log_std_min=cfg["log_std_min"],
            log_std_max=cfg["log_std_max"],
        ).to(device)
        self.critic = Critic(
            _encoder(shared_cnns, state_dim, cfg, shared_graph),
            action_dim=action_dim,
            hidden_dims=cfg["critic_hidden_dims"],
            layer_norm=bool(cfg["critic_layer_norm"]),
            dropout=cfg.get("critic_dropout"),
        ).to(device)
        self.critic_target = Critic(
            _encoder(target_cnns, state_dim, cfg, target_graph),
            action_dim=action_dim,
            hidden_dims=cfg["critic_hidden_dims"],
            layer_norm=bool(cfg["critic_layer_norm"]),
            dropout=cfg.get("critic_dropout"),
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            [float(np.log(float(cfg["init_temperature"])))],
            device=device, requires_grad=True,
        )
        self.target_entropy = -float(action_dim)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg["actor_lr"]),
            betas=(float(cfg["actor_beta"]), 0.999),
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg["critic_lr"]),
            betas=(float(cfg["critic_beta"]), 0.999),
        )
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha], lr=float(cfg["alpha_lr"]),
            betas=(float(cfg["alpha_beta"]), 0.999),
        )
        self._iter = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def act(
        self,
        pixels: Dict[str, torch.Tensor],
        state: torch.Tensor,
        graph: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            mu, _, _, _ = self.actor(
                pixels, state, graph, compute_pi=False, compute_log_pi=False,
            )
            return mu
        _, pi, _, _ = self.actor(pixels, state, graph, compute_log_pi=False)
        return pi

    def update(self, batch: Mapping[str, torch.Tensor]) -> UpdateMetrics:
        pix, pix_next = batch["pixel_obs"], batch["pixel_next_obs"]
        st, st_next = batch["state_obs"], batch["state_next_obs"]
        g = batch.get("graph_obs")
        g_next = batch.get("graph_next_obs")
        act = batch["act"]
        rew = batch["rew"].unsqueeze(-1)
        not_done = (1.0 - batch["term"]).unsqueeze(-1)

        with torch.no_grad():
            _, next_a, next_logpi, _ = self.actor(pix_next, st_next, g_next)
            tQ1, tQ2 = self.critic_target(pix_next, st_next, next_a, g_next)
            tV = torch.min(tQ1, tQ2) - self.log_alpha.exp().detach() * next_logpi
            target_Q = rew + not_done * self.gamma * tV

        Q1, Q2 = self.critic(pix, st, act, g)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss_val = alpha_loss_val = entropy_val = None
        if self._iter % self.actor_update_freq == 0:
            _, pi, log_pi, log_std = self.actor(pix, st, g, detach=True)
            aQ1, aQ2 = self.critic(pix, st, pi, g, detach=True)
            actor_loss = (self.log_alpha.exp().detach() * log_pi
                           - torch.min(aQ1, aQ2)).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optim.step()

            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss = (self.log_alpha.exp()
                           * (-log_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optim.step()

            entropy = (0.5 * log_std.size(-1) * (1.0 + np.log(2 * np.pi))
                        + log_std.sum(-1))
            actor_loss_val = float(actor_loss.item())
            alpha_loss_val = float(alpha_loss.item())
            entropy_val = float(entropy.mean().item())

        if self._iter % self.critic_target_update_freq == 0:
            self._soft_update()
        self._iter += 1

        return UpdateMetrics(
            critic_loss=float(critic_loss.item()),
            q1=float(Q1.mean().item()),
            q2=float(Q2.mean().item()),
            actor_loss=actor_loss_val,
            alpha=self.alpha,
            alpha_loss=alpha_loss_val,
            entropy_mean=entropy_val,
            log_alpha=float(self.log_alpha.item()),
            target_entropy=self.target_entropy,
        )

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.critic.Q1.parameters(),
                              self.critic_target.Q1.parameters()):
                tp.data.copy_(self.critic_tau * p.data
                               + (1 - self.critic_tau) * tp.data)
            for p, tp in zip(self.critic.Q2.parameters(),
                              self.critic_target.Q2.parameters()):
                tp.data.copy_(self.critic_tau * p.data
                               + (1 - self.critic_tau) * tp.data)
            for p, tp in zip(self.critic.encoder.parameters(),
                              self.critic_target.encoder.parameters()):
                tp.data.copy_(self.critic_encoder_tau * p.data
                               + (1 - self.critic_encoder_tau) * tp.data)
