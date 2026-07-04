"""Env builder for MS-HAB SAC. Mirrors mshab/envs/make.make_env.

When graph is enabled, obs_mode widens to ``rgb+depth+segmentation`` so the
graph pipeline can read segmentation (and, at eval time, RGB for the video
overlay) via ``env.unwrapped._last_obs``. The policy-facing obs is unaffected:
``FetchDepthObservationWrapper`` still exposes only depth + state.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import torch


def build_env(
    task: str, cfg: dict, *, is_eval: bool = False, seed: int = 0,
    graph_enabled: bool = False,
):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mshab.envs.wrappers import (
        FetchActionWrapper,
        FetchDepthObservationWrapper,
        FrameStack,
    )
    import mshab.envs  # noqa: F401
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    obs_mode = "rgb+depth+segmentation" if graph_enabled else "depth"
    num_envs = int(cfg["num_eval_envs"] if is_eval else cfg["num_envs"])
    image_size = int(cfg["image_size"])
    reconfiguration_freq = int(
        cfg["eval_reconfiguration_freq"] if is_eval else cfg["reconfiguration_freq"]
    ) or None
    horizon_key = "eval_max_episode_steps" if is_eval else "max_episode_steps"
    horizon = int(cfg[horizon_key])

    subtask = task.split("SubtaskTrain")[0].lower()
    rearrange_dir = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan = plan_data_from_file(
        rearrange_dir / "task_plans" / cfg["mshab_task"] / subtask
        / cfg["mshab_split"] / f"{cfg['mshab_obj']}.json"
    )

    env_kwargs = dict(
        task_plans=plan.plans,
        scene_builder_cls=plan.dataset,
        spawn_data_fp=(
            rearrange_dir / "spawn_data" / cfg["mshab_task"] / subtask
            / cfg["mshab_split"] / "spawn_data.pt"
        ),
        require_build_configs_repeated_equally_across_envs=False,
        robot_force_mult=float(cfg["robot_force_mult"]),
        robot_force_penalty_min=float(cfg["robot_force_penalty_min"]),
    )

    env = gym.make(
        task,
        max_episode_steps=horizon,
        obs_mode=obs_mode,
        reward_mode=cfg["reward_mode"],
        control_mode=cfg["control_mode"],
        render_mode="all",
        shader_dir="minimal",
        robot_uids="fetch",
        num_envs=num_envs,
        sim_backend=cfg["sim_backend"],
        reconfiguration_freq=reconfiguration_freq,
        sensor_configs=dict(width=image_size, height=image_size),
        **env_kwargs,
    )

    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(
        env, num_stack=int(cfg["frame_stack"]),
        stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
    )
    env = FetchActionWrapper(
        env,
        stationary_base=False,
        stationary_torso=False,
        stationary_head=bool(cfg.get("mshab_stationary_head", True)),
    )
    venv = ManiSkillVectorEnv(
        env, num_envs,
        max_episode_steps=horizon,
        ignore_terminations=True,
    )
    return _StatsWrapper(venv, max_episode_steps=horizon)


def adapt_obs(raw: Mapping, device: torch.device) -> Dict[str, torch.Tensor]:
    """Return ``{'state': [N, D], 'pixels': {'fetch_head_depth': ..., 'fetch_hand_depth': ...}}``."""
    state = raw["state"]
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(np.asarray(state), device=device)
    pixels = raw["pixels"]
    out_pixels: Dict[str, torch.Tensor] = {}
    for k, v in pixels.items():
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(np.asarray(v), device=device)
        out_pixels[k] = v.to(device=device, dtype=torch.float32)
    return {"state": state.to(device).float(), "pixels": out_pixels}


def action_box(env) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
    space = env.single_action_space
    return tuple(space.shape), np.asarray(space.low), np.asarray(space.high)


class _StatsWrapper:
    """Track episode return / length / success per env and expose queues.

    Replaces mshab's VectorRecordEpisodeStatistics, which depends on
    gymnasium.vector.VectorEnvWrapper -- removed in gymnasium 1.x.
    """

    def __init__(self, env, max_episode_steps: int):
        self.env = env
        self.max_episode_steps = int(max_episode_steps)
        self._device = env.unwrapped.device
        self._returns = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self._lengths = torch.zeros(env.num_envs, dtype=torch.int32, device=self._device)
        self._success_once = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self._success_at_end = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self.reset_queues()

    def reset_queues(self) -> None:
        self.return_queue = []
        self.length_queue = []
        self.success_once_queue = []
        self.success_at_end_queue = []

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._returns.zero_()
        self._lengths.zero_()
        self._success_once.zero_()
        self._success_at_end.zero_()
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._returns += rew
        self._lengths += 1
        s = info.get("success")
        if s is not None:
            s = s.to(dtype=torch.bool, device=self._device) if isinstance(s, torch.Tensor) \
                else torch.as_tensor(s, dtype=torch.bool, device=self._device)
            self._success_at_end = s
            self._success_once = self._success_once | s
        dones = term | trunc
        if dones.any():
            idx = torch.where(dones)[0]
            self.return_queue.extend(self._returns[idx].tolist())
            self.length_queue.extend(self._lengths[idx].tolist())
            self.success_once_queue.extend(self._success_once[idx].tolist())
            self.success_at_end_queue.extend(self._success_at_end[idx].tolist())
            self._returns[idx] = 0
            self._lengths[idx] = 0
            self._success_once[idx] = False
            self._success_at_end[idx] = False
        return obs, rew, term, trunc, info

    def __getattr__(self, name: str):
        return getattr(self.__dict__["env"], name)
