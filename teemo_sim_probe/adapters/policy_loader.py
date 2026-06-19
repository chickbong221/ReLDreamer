"""Load and run an MS-HAB PPO checkpoint, matching mshab/evaluate.py exactly.

Verified against the installed mshab source:
  * agent class: mshab.agents.ppo.Agent(sample_obs, single_act_shape)
  * eval action: agent.get_action(obs, deterministic=True)  -> bare tensor
  * checkpoint : torch.load(path)["agent"]
  * policy obs : depth + framestack dict produced by FetchDepthObservationWrapper
                 + FrameStack (NOT rgb, NOT the raw sensor_data dict)

The probe does not feed the policy a hand-built obs. The env is wrapped exactly
as mshab.envs.make.make_env wraps it, so ``venv.reset()`` already yields the
obs the policy expects. This loader just builds the net from that obs and
restores weights.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np


class PolicyHandle:
    """Uniform action interface over an mshab PPO Agent or a random fallback."""

    def __init__(self, fn: Callable[[Any], np.ndarray], kind: str):
        self._fn = fn
        self.kind = kind

    def act(self, obs: Any) -> np.ndarray:
        return self._fn(obs)


def find_checkpoint(ckpt_dir: str) -> str:
    """Locate the policy weights file inside a checkpoint dir.

    mshab checkpoints store ``policy.pt`` or ``latest.pt`` / ``best.pt``.
    """
    for name in ("policy.pt", "latest.pt", "best.pt", "ckpt.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    # any .pt as last resort
    for f in sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []:
        if f.endswith(".pt"):
            return os.path.join(ckpt_dir, f)
    return ""


def load_ppo_policy(
    ckpt_path_or_dir: str,
    venv,
    eval_obs: Any,
    device: str = "cuda",
) -> PolicyHandle:
    """Build mshab PPO Agent and restore weights, matching evaluate.py.

    Args:
        ckpt_path_or_dir: a .pt file or a dir containing one.
        venv: the wrapped vector env (for single_action_space).
        eval_obs: the policy-shaped obs from venv.reset() (depth + framestack).
        device: torch device.
    """
    ckpt = ckpt_path_or_dir
    if os.path.isdir(ckpt):
        ckpt = find_checkpoint(ckpt_path_or_dir)
    if not ckpt or not os.path.exists(ckpt):
        print(f"[policy] no checkpoint under {ckpt_path_or_dir!r} -> random actions")
        return _random_policy(venv)

    try:
        import torch
        from mshab.agents.ppo import Agent as PPOAgent

        # act_space from the UNWRAPPED env, exactly like evaluate.py.
        act_shape = venv.unwrapped.single_action_space.shape

        # to_tensor the obs (evaluate.py does to_tensor(eval_obs, dtype="float")).
        obs_t = _to_tensor_obs(eval_obs, device)

        policy = PPOAgent(obs_t, act_shape)
        policy.eval()
        state = torch.load(ckpt, map_location=device)
        policy.load_state_dict(state["agent"] if "agent" in state else state)
        policy.to(device)

        def _act(obs: Any) -> np.ndarray:
            o = _to_tensor_obs(obs, device)
            with torch.no_grad():
                a = policy.get_action(o, deterministic=True)  # bare tensor
            return a.detach().cpu().numpy()

        print(f"[policy] loaded mshab PPO Agent from {ckpt}")
        return PolicyHandle(_act, kind="ppo")

    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[policy] could not load mshab PPO Agent ({exc}); random actions")
        traceback.print_exc()
        return _random_policy(venv)


def _to_tensor_obs(obs, device):
    """Recursively move a (possibly nested) obs dict to float tensors on device."""
    import torch
    if isinstance(obs, dict):
        return {k: _to_tensor_obs(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device, dtype=torch.float)
    return torch.as_tensor(np.asarray(obs), device=device, dtype=torch.float)


def _random_policy(venv) -> PolicyHandle:
    space = venv.action_space

    def _act(obs):
        return space.sample()

    return PolicyHandle(_act, kind="random")
