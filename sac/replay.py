"""Episode-structured replay buffer ported from mshab/agents/sac/replay.py.

Pixels stored as uint16 with frame-stack materialized at sample time from
neighbor timesteps. State stored per-step. Graph tensors (if enabled) stored
per-step in their native dtypes.

Any batch containing a truncated env is dropped: matches mshab's stability
trick for continuous-task envs.
"""

from __future__ import annotations

from typing import Dict, Optional

from gymnasium import spaces

import numpy as np


class PixelStateBatchReplayBuffer:
    def __init__(
        self,
        pixels_obs_space: spaces.Dict,
        state_obs_dim: int,
        act_dim: int,
        size: int,
        horizon: int,
        num_envs: int,
        graph_shapes: Optional[Dict[str, tuple]] = None,
        graph_dtypes: Optional[Dict[str, np.dtype]] = None,
    ):
        frame_stacks = [pixels_obs_space[k].shape[0] for k in pixels_obs_space]
        assert frame_stacks.count(frame_stacks[0]) == len(frame_stacks), (
            f"replay expects same frame_stack across pixel keys; got {frame_stacks}"
        )
        frame_stack = frame_stacks[0]
        assert size % horizon == 0, "size must be divisible by horizon"
        num_episodes = size // horizon
        assert num_episodes % num_envs == 0, (
            "num_episodes must be divisible by num_envs"
        )

        self.pixel_obs_buf: Dict[str, np.ndarray] = {}
        self.transpose_order: Dict[str, tuple] = {}
        for k, space in pixels_obs_space.items():
            self.pixel_obs_buf[k] = np.zeros(
                [num_episodes, horizon + frame_stack + 1, *space.shape[1:]],
                dtype=np.uint16,
            )
            to = list(range(len(space.shape) + 1))
            to[0], to[1] = 1, 0
            self.transpose_order[k] = tuple(to)

        self.state_obs_buf = np.zeros([num_episodes, horizon, state_obs_dim], dtype=np.float32)
        self.state_next_obs_buf = np.zeros([num_episodes, horizon, state_obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([num_episodes, horizon, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([num_episodes, horizon], dtype=np.float32)
        self.done_buf = np.zeros([num_episodes, horizon], dtype=np.float32)

        self.graph_obs_buf: Dict[str, np.ndarray] = {}
        self.graph_next_obs_buf: Dict[str, np.ndarray] = {}
        if graph_shapes:
            assert graph_dtypes is not None
            for k, shape in graph_shapes.items():
                dt = graph_dtypes[k]
                self.graph_obs_buf[k] = np.zeros([num_episodes, horizon, *shape], dtype=dt)
                self.graph_next_obs_buf[k] = np.zeros([num_episodes, horizon, *shape], dtype=dt)

        self.idx_to_coord = np.zeros([size, 2], dtype=np.uint32)

        self.max_size = size
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.num_envs = num_envs
        self.frame_stack = frame_stack

        self.current_size = 0
        self.batch_start_episode = 0
        self.step_num = 0

    def __len__(self) -> int:
        return self.current_size

    def store_batch(
        self,
        pixel_obs: Dict[str, np.ndarray],
        pixel_next_obs: Dict[str, np.ndarray],
        state_obs: np.ndarray,
        state_next_obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        term: np.ndarray,
        graph_obs: Optional[Dict[str, np.ndarray]] = None,
        graph_next_obs: Optional[Dict[str, np.ndarray]] = None,
        trunc_any: bool = False,
    ) -> None:
        if trunc_any:
            return
        bs, be = self.batch_start_episode, self.batch_start_episode + self.num_envs
        sn, fs = self.step_num, self.frame_stack

        for k in self.pixel_obs_buf:
            self.pixel_obs_buf[k][bs:be, sn : sn + fs] = pixel_obs[k]
            self.pixel_obs_buf[k][bs:be, sn + 1 : sn + fs + 1] = pixel_next_obs[k]

        self.state_obs_buf[bs:be, sn] = state_obs
        self.state_next_obs_buf[bs:be, sn] = state_next_obs
        self.acts_buf[bs:be, sn] = act
        self.rews_buf[bs:be, sn] = rew
        self.done_buf[bs:be, sn] = term

        for k in self.graph_obs_buf:
            self.graph_obs_buf[k][bs:be, sn] = graph_obs[k]
            self.graph_next_obs_buf[k][bs:be, sn] = graph_next_obs[k]

        if self.current_size < self.max_size:
            self.idx_to_coord[self.current_size : self.current_size + self.num_envs] = (
                np.stack([np.arange(bs, be), np.repeat(sn, be - bs)], axis=-1)
            )

        self.step_num += 1
        self.current_size = min(self.current_size + self.num_envs, self.max_size)
        if self.step_num == self.horizon:
            self.step_num = 0
            self.batch_start_episode += self.num_envs
        if self.batch_start_episode == self.num_episodes:
            self.batch_start_episode = 0

    def sample_batch(self, batch_size: int) -> Dict[str, object]:
        idxs = np.random.randint(0, self.current_size, size=batch_size)
        coords = self.idx_to_coord[idxs]
        ep_nums, step_nums = coords[:, 0], coords[:, 1]

        fs_step_nums = np.stack([step_nums + i for i in range(self.frame_stack)])
        pixel_obs: Dict[str, np.ndarray] = {}
        pixel_next_obs: Dict[str, np.ndarray] = {}
        for k in self.pixel_obs_buf:
            pixel_obs[k] = self.pixel_obs_buf[k][ep_nums, fs_step_nums].transpose(
                self.transpose_order[k]
            )
            pixel_next_obs[k] = self.pixel_obs_buf[k][ep_nums, fs_step_nums + 1].transpose(
                self.transpose_order[k]
            )

        batch: Dict[str, object] = dict(
            pixel_obs=pixel_obs,
            pixel_next_obs=pixel_next_obs,
            state_obs=self.state_obs_buf[ep_nums, step_nums],
            state_next_obs=self.state_next_obs_buf[ep_nums, step_nums],
            act=self.acts_buf[ep_nums, step_nums],
            rew=self.rews_buf[ep_nums, step_nums],
            term=self.done_buf[ep_nums, step_nums],
        )
        if self.graph_obs_buf:
            batch["graph_obs"] = {
                k: self.graph_obs_buf[k][ep_nums, step_nums] for k in self.graph_obs_buf
            }
            batch["graph_next_obs"] = {
                k: self.graph_next_obs_buf[k][ep_nums, step_nums] for k in self.graph_next_obs_buf
            }
        return batch
