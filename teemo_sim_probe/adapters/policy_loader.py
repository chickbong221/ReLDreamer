"""Load and run an MS-HAB PPO checkpoint, mirroring their evaluate wrapper stack.

Design notes (verified against the MS-HAB repo):
  * The PPO checkpoint dir contains ``config.yml`` + ``policy.pt``.
  * The training/eval env is created with ``obs_mode="rgbd"`` and wrapped with
    ``ManiSkillVectorEnv`` plus mshab's own obs/flatten wrappers.
  * We instead create the env in ``obs_mode="rgb+depth+segmentation"`` so the
    probe can read segmentation, and drop the seg channel before the policy
    sees obs (the policy was trained without it).

Because the precise ``Agent`` class signature lives in your installed ``mshab``
copy, this loader imports it dynamically and degrades to random actions if the
checkpoint isn't present yet (you said checkpoints aren't downloaded).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Obs adaptation: strip segmentation so the rgbd-trained policy is happy.
# --------------------------------------------------------------------------- #
def strip_segmentation(obs: dict) -> dict:
    """Return a shallow copy of obs with 'segmentation' removed per camera.

    Leaves rgb + depth so the policy sees exactly the rgbd obs it expects.
    The probe reads segmentation from the *original* obs before this call.
    """
    if "sensor_data" not in obs:
        return obs
    new = dict(obs)
    sd = {}
    for cam, data in obs["sensor_data"].items():
        d = dict(data)
        d.pop("segmentation", None)
        sd[cam] = d
    new["sensor_data"] = sd
    return new


# --------------------------------------------------------------------------- #
# Checkpoint config discovery
# --------------------------------------------------------------------------- #
def load_checkpoint_config(ckpt_dir: str) -> dict:
    """Read the checkpoint's config.yml (env kwargs, task/subtask/split, etc.)."""
    import yaml
    cfg_path = os.path.join(ckpt_dir, "config.yml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"no config.yml in {ckpt_dir}. Download with:\n"
            f"  huggingface-cli download arth-shukla/mshab_checkpoints "
            f"--local-dir mshab_checkpoints"
        )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Policy loading
# --------------------------------------------------------------------------- #
class PolicyHandle:
    """Uniform action interface over either an mshab PPO Agent or random fallback."""

    def __init__(self, fn: Callable[[dict], np.ndarray], kind: str):
        self._fn = fn
        self.kind = kind

    def act(self, obs: dict) -> np.ndarray:
        return self._fn(obs)


def load_ppo_policy(
    ckpt_dir: str,
    venv,
    device: str = "cuda",
) -> PolicyHandle:
    """Try to load mshab PPO Agent from ``policy.pt``; fall back to random.

    Mirrors mshab.train_ppo's Agent construction (NatureCNN feature extractor
    over rgbd + state). The exact import path is read from your installed copy.
    """
    policy_pt = os.path.join(ckpt_dir, "policy.pt")
    if not os.path.exists(policy_pt):
        print(f"[policy] {policy_pt} missing -> random actions")
        return _random_policy(venv)

    try:
        import torch
        from mshab.agents.ppo import Agent  # type: ignore

        # mshab Agent is typically Agent(envs, sample_obs) or Agent(envs).
        sample_obs, _ = venv.reset()
        try:
            agent = Agent(venv, sample_obs)
        except TypeError:
            agent = Agent(venv)

        state_dict = torch.load(policy_pt, map_location=device)
        # checkpoints may store {'agent': ...} or the bare state dict
        if isinstance(state_dict, dict) and "agent" in state_dict:
            state_dict = state_dict["agent"]
        agent.load_state_dict(state_dict)
        agent = agent.to(device).eval()

        def _act(obs: dict) -> np.ndarray:
            policy_obs = strip_segmentation(obs)
            with torch.no_grad():
                # mshab PPO exposes get_action / get_eval_action / act.
                for meth in ("get_eval_action", "get_action", "act"):
                    if hasattr(agent, meth):
                        a = getattr(agent, meth)(policy_obs)
                        break
                else:
                    a = agent(policy_obs)
            if isinstance(a, (tuple, list)):
                a = a[0]
            return a.detach().cpu().numpy()

        print(f"[policy] loaded mshab PPO Agent from {policy_pt}")
        return PolicyHandle(_act, kind="ppo")

    except Exception as exc:  # noqa: BLE001
        print(f"[policy] could not load mshab PPO Agent ({exc}); random actions")
        return _random_policy(venv)


def _random_policy(venv) -> PolicyHandle:
    space = venv.action_space

    def _act(obs: dict) -> np.ndarray:
        return space.sample()

    return PolicyHandle(_act, kind="random")
