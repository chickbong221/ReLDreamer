"""Flat circular replay buffer with dict-obs support.

Storage layout: per-env circular slot index. The buffer holds
(obs, action, reward, next_obs, done) tuples; obs / next_obs are dicts of
per-key tensors. Sampling draws B independent (slot, env) indices uniformly.

Capacity is measured in transitions. ``per_env_buffer_size`` is
``buffer_size // num_envs`` so each parallel env owns an equal slice.

Lifted with minor cleanup from ManiSkill's `examples/baselines/sac/sac_rgbd.py`
(`DictArray` + `ReplayBuffer`); the user's design call was to stay close to
that reference for drop-in actor/critic compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch


def _torch_dtype_for(dtype) -> torch.dtype:
    if dtype in (np.float32, np.float64, torch.float32, torch.float64):
        return torch.float32
    if dtype in (np.uint8, torch.uint8):
        return torch.uint8
    if dtype in (np.int16, torch.int16):
        return torch.int16
    if dtype in (np.int32, torch.int32):
        return torch.int32
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.float32


class DictArray:
    """Tensor-of-dicts container, keyed by str -> Tensor of shape (T, N, *)."""

    def __init__(self, buffer_shape, spec: Mapping[str, "TensorSpec"],
                  device=None):
        self.buffer_shape = tuple(buffer_shape)
        self.data: Dict[str, torch.Tensor] = {}
        for k, v in spec.items():
            self.data[k] = torch.zeros(
                self.buffer_shape + tuple(v.shape),
                dtype=_torch_dtype_for(v.dtype),
                device=device,
            )

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value: Mapping[str, torch.Tensor]):
        for k, v in value.items():
            self.data[k][index] = v

    def keys(self):
        return self.data.keys()


@dataclass
class TensorSpec:
    shape: tuple
    dtype: object


@dataclass
class ReplayBufferSample:
    obs: Dict[str, torch.Tensor]
    next_obs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Flat circular buffer for SAC. Per-env stripes; uniform sampling."""

    def __init__(
        self,
        obs_spec: Mapping[str, TensorSpec],
        action_shape,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device,
        sample_device: torch.device,
    ):
        self.num_envs = int(num_envs)
        self.per_env_buffer_size = int(buffer_size) // self.num_envs
        if self.per_env_buffer_size < 1:
            raise ValueError(
                f"buffer_size={buffer_size} too small for num_envs={num_envs}")
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.pos = 0
        self.full = False

        shape = (self.per_env_buffer_size, self.num_envs)
        self.obs = DictArray(shape, obs_spec, device=storage_device)
        self.next_obs = DictArray(shape, obs_spec, device=storage_device)
        self.actions = torch.zeros(
            shape + tuple(action_shape), dtype=torch.float32, device=storage_device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=storage_device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=storage_device)

    def __len__(self) -> int:
        return (self.per_env_buffer_size if self.full else self.pos) * self.num_envs

    def _to_storage(self, t: torch.Tensor) -> torch.Tensor:
        if self.storage_device == torch.device("cpu"):
            return t.detach().cpu()
        return t.detach().to(self.storage_device, non_blocking=True)

    def add(
        self,
        obs: Mapping[str, torch.Tensor],
        next_obs: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        slot = self.pos
        for k, v in obs.items():
            self.obs.data[k][slot] = self._to_storage(v)
        for k, v in next_obs.items():
            self.next_obs.data[k][slot] = self._to_storage(v)
        self.actions[slot] = self._to_storage(action)
        self.rewards[slot] = self._to_storage(reward.float())
        self.dones[slot] = self._to_storage(done.float())

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        slot_hi = self.per_env_buffer_size if self.full else self.pos
        if slot_hi == 0:
            raise RuntimeError("ReplayBuffer.sample called before any add()")
        slot_idx = torch.randint(0, slot_hi, size=(batch_size,))
        env_idx = torch.randint(0, self.num_envs, size=(batch_size,))

        def _gather(d: DictArray) -> Dict[str, torch.Tensor]:
            return {k: v[slot_idx, env_idx].to(self.sample_device,
                                                non_blocking=True)
                    for k, v in d.data.items()}

        return ReplayBufferSample(
            obs=_gather(self.obs),
            next_obs=_gather(self.next_obs),
            actions=self.actions[slot_idx, env_idx].to(self.sample_device,
                                                       non_blocking=True),
            rewards=self.rewards[slot_idx, env_idx].to(self.sample_device,
                                                       non_blocking=True),
            dones=self.dones[slot_idx, env_idx].to(self.sample_device,
                                                    non_blocking=True),
        )


def build_obs_spec(sample_obs: Mapping[str, torch.Tensor]) -> Dict[str, TensorSpec]:
    """Infer a TensorSpec dict from a sample batched obs (axis 0 = num_envs)."""
    spec: Dict[str, TensorSpec] = {}
    for k, v in sample_obs.items():
        spec[k] = TensorSpec(shape=tuple(v.shape[1:]), dtype=v.dtype)
    return spec
