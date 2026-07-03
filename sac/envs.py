"""Env builder for ManiSkill 3 and MS-HAB, returning a ManiSkillVectorEnv.

Mirrors the wrapper stacks used in:
  - ManiSkill/examples/baselines/sac/sac_rgbd.py       (state / rgb modes)
  - mshab/mshab/envs/make.py + mshab/mshab/train_sac.py (depth mode)

The returned env is a ``ManiSkillVectorEnv`` exposing ``single_observation_space``
and ``single_action_space`` so the SAC agent / replay buffer can size their
buffers off of it. The raw observation dict still uses ManiSkill key names
(``rgb`` / ``fetch_head_depth`` / ``fetch_hand_depth`` / ``state``); call
``adapt_obs`` to collapse it to the agent-friendly schema:

    state-only : {'state': [N, D]}
    rgb        : {'state': [N, D], 'rgb':   [N, H, W, C]}    (uint8, NHWC)
    depth      : {'state': [N, D], 'depth': [N, H, W, C]}    (float, NHWC)
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import torch


def build_env(
    task: str, cfg: dict, *, is_eval: bool = False, seed: int = 0,
    graph_enabled: bool = False,
):
    """Construct a ManiSkillVectorEnv for the requested obs_mode / suite.

    ``task`` is the ManiSkill env id (e.g. ``PickCube-v1`` or
    ``PickSubtaskTrain-v0``). ``cfg`` is the ``env.maniskill`` config block.

    When ``graph_enabled`` is True the underlying env's obs_mode is widened to
    include segmentation so the graph pipeline can read masks via
    ``env.unwrapped.get_obs()``; the policy-facing obs is unaffected.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers tasks
    from mani_skill.utils.wrappers.flatten import (
        FlattenActionSpaceWrapper,
        FlattenRGBDObservationWrapper,
    )
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    obs_mode = str(cfg["obs_mode"])
    underlying_obs_mode = (
        "rgb+depth+segmentation" if graph_enabled else obs_mode
    )
    num_envs = int(cfg["num_eval_envs"] if is_eval else cfg["num_envs"])
    image_size = int(cfg["image_size"])
    sim_backend = str(cfg["sim_backend"])
    reconfiguration_freq = int(
        cfg["eval_reconfiguration_freq"] if is_eval else cfg["reconfiguration_freq"]
    ) or None
    mshab_active = str(cfg.get("mshab_task", "none")).lower() != "none"

    make_kwargs: Dict[str, object] = dict(
        id=task,
        obs_mode=underlying_obs_mode,
        num_envs=num_envs,
        sim_backend=sim_backend,
        sensor_configs=dict(width=image_size, height=image_size),
        reconfiguration_freq=reconfiguration_freq,
        render_mode="rgb_array" if (obs_mode != "state" or is_eval) else None,
    )
    if cfg.get("control_mode"):
        make_kwargs["control_mode"] = cfg["control_mode"]
    if cfg.get("reward_mode"):
        make_kwargs["reward_mode"] = cfg["reward_mode"]

    # MS-HAB task-plan setup (only used when the user picked a *SubtaskTrain
    # env and configured a `mshab_task` in the env block).
    if mshab_active:
        import mshab.envs  # noqa: F401  registers mshab envs
        from mani_skill import ASSET_DIR
        from mshab.envs.planner import plan_data_from_file

        subtask = task.split("SubtaskTrain")[0].lower()
        rearrange_dir = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
        plan = plan_data_from_file(
            rearrange_dir / "task_plans" / cfg["mshab_task"] / subtask
            / cfg["mshab_split"] / f"{cfg['mshab_obj']}.json"
        )
        make_kwargs["task_plans"] = plan.plans
        make_kwargs["scene_builder_cls"] = plan.dataset
        make_kwargs["spawn_data_fp"] = (
            rearrange_dir / "spawn_data" / cfg["mshab_task"] / subtask
            / cfg["mshab_split"] / "spawn_data.pt"
        )
        make_kwargs["require_build_configs_repeated_equally_across_envs"] = False
        make_kwargs.setdefault("shader_dir", "minimal")
        horizon_key = "eval_max_episode_steps" if is_eval else "max_episode_steps"
        horizon = int(cfg[horizon_key]) or 0
        if horizon > 0:
            make_kwargs["max_episode_steps"] = horizon

    env = gym.make(**make_kwargs)

    if obs_mode == "rgb":
        env = FlattenRGBDObservationWrapper(
            env, rgb=True, depth=False, state=bool(cfg["include_state"]),
        )
    elif obs_mode == "depth" and mshab_active:
        from mshab.envs.wrappers import (
            FetchActionWrapper,
            FetchDepthObservationWrapper,
        )
        env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
        env = FetchActionWrapper(
            env,
            stationary_base=False,
            stationary_torso=False,
            stationary_head=bool(cfg.get("mshab_stationary_head", True)),
        )
    # state-only path: no obs wrapper needed.

    import gymnasium as gym
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    env = ManiSkillVectorEnv(
        env, num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )
    return env


# --------------------------------------------------------------------------- #
# Obs adaptation: raw ManiSkill dict -> agent-friendly dict.
# --------------------------------------------------------------------------- #
def adapt_obs(raw: Mapping, obs_mode: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Return ``{'state': ..., ['rgb'|'depth']: ...}``.

    - ``state-only``  : flattens any agent/extra subdicts into one [N, D] tensor.
    - ``rgb``         : keeps ``raw['rgb']`` as uint8 NHWC, fetches state directly.
    - ``depth`` (mshab): concats fetch_head_depth + fetch_hand_depth into one
                        float NHWC ``depth`` tensor.
    """
    out: Dict[str, torch.Tensor] = {}

    if obs_mode == "state":
        out["state"] = _flatten_state(raw, device)
        return out

    state = raw["state"] if "state" in raw else _flatten_state(raw, device)
    out["state"] = state.to(device).float()

    if obs_mode == "rgb":
        rgb = raw["rgb"]
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.as_tensor(np.asarray(rgb), device=device)
        out["rgb"] = rgb.to(device)
    elif obs_mode == "depth":
        # FetchDepthObservationWrapper hands us [N, 1, H, W] channel-first.
        # Permute to NHWC and concat head + hand cameras on the channel axis.
        head = raw["fetch_head_depth"]
        hand = raw["fetch_hand_depth"]
        if not isinstance(head, torch.Tensor):
            head = torch.as_tensor(np.asarray(head), device=device)
            hand = torch.as_tensor(np.asarray(hand), device=device)
        head = head.to(device).float().permute(0, 2, 3, 1)
        hand = hand.to(device).float().permute(0, 2, 3, 1)
        out["depth"] = torch.cat([head, hand], dim=-1)
    else:
        raise ValueError(f"Unknown obs_mode {obs_mode!r}")
    return out


def _flatten_state(raw: Mapping, device: torch.device) -> torch.Tensor:
    """Concat agent + extra proprio subdicts into [N, D] (state obs mode)."""
    if isinstance(raw, torch.Tensor):
        return raw.to(device).float()
    if "state" in raw and isinstance(raw["state"], torch.Tensor):
        return raw["state"].to(device).float()
    parts = []
    for k in ("agent", "extra"):
        sub = raw.get(k, {})
        if isinstance(sub, dict):
            for v in sub.values():
                v = v if isinstance(v, torch.Tensor) else torch.as_tensor(
                    np.asarray(v), device=device)
                parts.append(v.reshape(v.shape[0], -1))
    if not parts:
        raise ValueError(f"No state found in obs keys: {list(raw.keys())}")
    return torch.cat(parts, dim=-1).to(device).float()


def action_box(env) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
    """Return (action_shape, low, high) from a ManiSkillVectorEnv."""
    space = env.single_action_space
    return tuple(space.shape), np.asarray(space.low), np.asarray(space.high)
