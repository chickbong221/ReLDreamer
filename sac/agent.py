"""SAC agent: squashed-Gaussian actor + twin Q-critics + learned alpha.

One implementation that adapts to three obs modes:
    - state-only      (obs = {'state': [N, D]})
    - pixel + state   (obs contains 'rgb' or 'depth' alongside 'state')

When a pixel key is present, the actor builds a PlainConv encoder and the
critics share it. State-only mode skips the encoder entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import GraphEncoder, has_graph_obs
from .nets import (
    EncoderObsWrapper,
    MultiObsEncoder,
    PlainConv,
    infer_image_channels,
    infer_image_size,
    make_mlp,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# --------------------------------------------------------------------------- #
# Actor / Critic
# --------------------------------------------------------------------------- #
class Actor(nn.Module):
    """Squashed-Gaussian actor. Encoder is optional; state path is always on."""

    def __init__(
        self,
        action_shape,
        state_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        encoder: Optional[EncoderObsWrapper],
        mlp_hidden=(512, 256),
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = state_dim + (encoder.out_dim if encoder is not None else 0)
        self.trunk = make_mlp(in_dim, list(mlp_hidden), last_act=True)
        self.fc_mean = nn.Linear(mlp_hidden[-1], int(np.prod(action_shape)))
        self.fc_logstd = nn.Linear(mlp_hidden[-1], int(np.prod(action_shape)))
        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def _embed(self, obs: Mapping[str, torch.Tensor], detach_encoder: bool):
        if self.encoder is None:
            return obs["state"], None
        visual = self.encoder(obs)
        if detach_encoder:
            visual = visual.detach()
        return torch.cat([visual, obs["state"]], dim=1), visual

    def forward(self, obs: Mapping[str, torch.Tensor], detach_encoder: bool = False):
        x, visual = self._embed(obs, detach_encoder)
        h = self.trunk(x)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std, visual

    @torch.no_grad()
    def get_eval_action(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mean, _, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, obs: Mapping[str, torch.Tensor],
                    detach_encoder: bool = False):
        mean, log_std, visual = self(obs, detach_encoder)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action, visual


class SoftQNetwork(nn.Module):
    """Q(s, a). If a shared encoder is provided, concats visual features."""

    def __init__(
        self,
        action_shape,
        state_dim: int,
        encoder: Optional[EncoderObsWrapper],
        mlp_hidden=(512, 256),
    ):
        super().__init__()
        self.encoder = encoder
        act_dim = int(np.prod(action_shape))
        in_dim = state_dim + act_dim + (encoder.out_dim if encoder is not None else 0)
        self.mlp = make_mlp(in_dim, list(mlp_hidden) + [1], last_act=False)

    def forward(
        self,
        obs: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        visual: Optional[torch.Tensor] = None,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        if self.encoder is not None:
            if visual is None:
                visual = self.encoder(obs)
            if detach_encoder:
                visual = visual.detach()
            x = torch.cat([visual, obs["state"], action], dim=1)
        else:
            x = torch.cat([obs["state"], action], dim=1)
        return self.mlp(x)


# --------------------------------------------------------------------------- #
# SAC agent
# --------------------------------------------------------------------------- #
@dataclass
class UpdateMetrics:
    qf_loss: float
    qf1_value: float
    qf2_value: float
    actor_loss: float
    alpha: float
    alpha_loss: float


class SACAgent:
    """SAC training wrapper.

    Owns the actor, twin critics + targets, optimizers, and (optionally) the
    learned log-alpha. Update logic mirrors ManiSkill's sac_rgbd baseline.
    """

    def __init__(
        self,
        sample_obs: Dict[str, torch.Tensor],   # one batched obs from env.reset
        action_shape,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: dict,
        device: torch.device,
    ):
        self.device = device
        self.cfg = cfg
        self.gamma = float(cfg["gamma"])
        self.tau = float(cfg["tau"])
        self.policy_frequency = int(cfg["policy_frequency"])
        self.target_network_frequency = int(cfg["target_network_frequency"])
        self.detach_critic_encoder = bool(cfg["encoder"]["detach_critic_encoder"])

        # --- pick encoder based on obs keys ----------------------------------
        rgb_key = "rgb" if "rgb" in sample_obs else None
        depth_key = "depth" if "depth" in sample_obs else None
        has_pixels = rgb_key is not None or depth_key is not None
        has_graph = has_graph_obs(sample_obs)

        sub_encoders = {}
        if has_pixels:
            in_c = infer_image_channels(sample_obs, rgb_key, depth_key)
            img_size = infer_image_size(sample_obs, rgb_key, depth_key)
            sub_encoders["pixels"] = EncoderObsWrapper(
                PlainConv(in_channels=in_c,
                          out_dim=int(cfg["encoder"]["out_dim"]),
                          image_size=img_size),
                rgb_key=rgb_key, depth_key=depth_key,
            )
        if has_graph:
            gcfg = cfg["encoder"].get("graph", {})
            sub_encoders["graph"] = GraphEncoder(
                node_vocab_size=int(gcfg["node_vocab_size"]),
                edge_vocab_size=int(gcfg["edge_vocab_size"]),
                embed_dim=int(gcfg.get("embed_dim", 64)),
                hidden_dim=int(gcfg.get("hidden_dim", 256)),
                out_dim=int(gcfg.get("out_dim", 128)),
                num_layers=int(gcfg.get("num_layers", 2)),
            )

        encoder = MultiObsEncoder(sub_encoders) if sub_encoders else None

        state_dim = int(sample_obs["state"].shape[1])
        mlp_hidden = tuple(cfg["encoder"]["mlp_hidden"])

        self.actor = Actor(
            action_shape, state_dim, action_low, action_high,
            encoder=encoder, mlp_hidden=mlp_hidden,
        ).to(device)

        # Critics share the actor's encoder (None in state-only mode).
        crit_encoder = self.actor.encoder
        self.qf1 = SoftQNetwork(action_shape, state_dim, crit_encoder, mlp_hidden).to(device)
        self.qf2 = SoftQNetwork(action_shape, state_dim, crit_encoder, mlp_hidden).to(device)
        self.qf1_target = SoftQNetwork(action_shape, state_dim, crit_encoder, mlp_hidden).to(device)
        self.qf2_target = SoftQNetwork(action_shape, state_dim, crit_encoder, mlp_hidden).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Optimizers: critics carry the shared-encoder grads; actor optimizer
        # only updates the actor head (the encoder receives grads through the
        # critic optimizer when shared, matching the ManiSkill baseline).
        q_params = list(self.qf1.mlp.parameters()) + list(self.qf2.mlp.parameters())
        if crit_encoder is not None:
            q_params += list(crit_encoder.parameters())
        self.q_optim = torch.optim.Adam(q_params, lr=float(cfg["q_lr"]))
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg["policy_lr"]),
        )

        # Entropy temperature.
        self.autotune = bool(cfg["autotune"])
        target_entropy_cfg = cfg["target_entropy"]
        if target_entropy_cfg == "auto":
            self.target_entropy = -float(np.prod(action_shape))
        else:
            self.target_entropy = float(target_entropy_cfg)
        if self.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=float(cfg["alpha_lr"]),
            )
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.log_alpha = None
            self.alpha_optim = None
            self.alpha = float(cfg["alpha"])

        self._global_update = 0

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.actor.get_eval_action(obs)
        action, _, _, _ = self.actor.get_action(obs)
        return action

    # ------------------------------------------------------------------ #
    # Update step
    # ------------------------------------------------------------------ #
    def update(self, batch) -> UpdateMetrics:
        obs, next_obs = batch.obs, batch.next_obs
        actions = batch.actions
        rewards = batch.rewards.view(-1)
        dones = batch.dones.view(-1)

        # ---- critic ---------------------------------------------------- #
        with torch.no_grad():
            next_a, next_logpi, _, next_visual = self.actor.get_action(next_obs)
            qf1_next = self.qf1_target(next_obs, next_a, next_visual).view(-1)
            qf2_next = self.qf2_target(next_obs, next_a, next_visual).view(-1)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_logpi.view(-1)
            target_q = rewards + (1.0 - dones) * self.gamma * min_qf_next

        visual = self.actor.encoder(obs) if self.actor.encoder is not None else None
        qf1_a = self.qf1(obs, actions, visual).view(-1)
        qf2_a = self.qf2(obs, actions, visual).view(-1)
        qf1_loss = F.mse_loss(qf1_a, target_q)
        qf2_loss = F.mse_loss(qf2_a, target_q)
        qf_loss = qf1_loss + qf2_loss

        self.q_optim.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optim.step()

        # ---- actor + alpha (delayed) ---------------------------------- #
        actor_loss_val = 0.0
        alpha_loss_val = 0.0
        if self._global_update % self.policy_frequency == 0:
            pi, log_pi, _, visual_pi = self.actor.get_action(obs)
            qf1_pi = self.qf1(obs, pi, visual_pi,
                              detach_encoder=self.detach_critic_encoder).view(-1)
            qf2_pi = self.qf2(obs, pi, visual_pi,
                              detach_encoder=self.detach_critic_encoder).view(-1)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = (self.alpha * log_pi.view(-1) - min_qf_pi).mean()

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optim.step()
            actor_loss_val = float(actor_loss.item())

            if self.autotune:
                with torch.no_grad():
                    _, log_pi2, _, _ = self.actor.get_action(obs)
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi2.view(-1) + self.target_entropy)
                ).mean()
                self.alpha_optim.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = float(self.log_alpha.exp().item())
                alpha_loss_val = float(alpha_loss.item())

        # ---- target soft update --------------------------------------- #
        if self._global_update % self.target_network_frequency == 0:
            with torch.no_grad():
                for p, tp in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        self._global_update += 1

        return UpdateMetrics(
            qf_loss=float(qf_loss.item() / 2.0),
            qf1_value=float(qf1_a.mean().item()),
            qf2_value=float(qf2_a.mean().item()),
            actor_loss=actor_loss_val,
            alpha=self.alpha,
            alpha_loss=alpha_loss_val,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        d = dict(
            actor=self.actor.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            qf1_target=self.qf1_target.state_dict(),
            qf2_target=self.qf2_target.state_dict(),
            q_optim=self.q_optim.state_dict(),
            actor_optim=self.actor_optim.state_dict(),
            global_update=self._global_update,
        )
        if self.autotune:
            d["log_alpha"] = self.log_alpha.detach().cpu()
            d["alpha_optim"] = self.alpha_optim.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.qf1.load_state_dict(d["qf1"])
        self.qf2.load_state_dict(d["qf2"])
        self.qf1_target.load_state_dict(d["qf1_target"])
        self.qf2_target.load_state_dict(d["qf2_target"])
        self.q_optim.load_state_dict(d["q_optim"])
        self.actor_optim.load_state_dict(d["actor_optim"])
        self._global_update = int(d.get("global_update", 0))
        if self.autotune and "log_alpha" in d:
            with torch.no_grad():
                self.log_alpha.copy_(d["log_alpha"].to(self.device))
            self.alpha = float(self.log_alpha.exp().item())
            self.alpha_optim.load_state_dict(d["alpha_optim"])
