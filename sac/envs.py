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
    from mshab.envs.wrappers.vector import VectorRecordEpisodeStatistics
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
    venv = VectorRecordEpisodeStatistics(
        venv, max_episode_steps=horizon,
    )
    return venv


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
        out_pixels[k] = v.to(device)
    return {"state": state.to(device).float(), "pixels": out_pixels}


def action_box(env) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
    space = env.single_action_space
    return tuple(space.shape), np.asarray(space.low), np.asarray(space.high)
