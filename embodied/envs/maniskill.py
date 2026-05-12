"""
ManiSkill GPU-vectorized environment wrapper for DreamerV3.

Design mirrors TD-MPC2's envs/maniskill.py + envs/wrappers/pixels.py:
  - One instance wraps N parallel GPU envs via ManiSkillVectorEnv
  - Returns batched numpy dicts [N, ...] from step()
  - Driver (batched=True) splits these into N per-worker replay transitions
  - Frame stacking for RGB mode lives entirely on GPU (no CPU transfers)

Key difference from TD-MPC2 pixel wrapper:
  - TD-MPC2 returns channels-first [N, C*F, H, W] (PyTorch convention)
  - DreamerV3 Encoder expects channels-last [H, W, C*F] (JAX/TF convention)
  - Images must be uint8 (Encoder asserts this and normalises internally)

Usage:
  python dreamerv3/main.py --configs maniskill_state task=maniskill_PickCube-v1
  python dreamerv3/main.py --configs maniskill_rgb   task=maniskill_PickCube-v1
"""

import numpy as np
import embodied
import elements


class ManiSkill(embodied.Env):

  def __init__(
      self,
      task,
      num_envs=32,
      obs_mode='state',
      image_size=64,
      sim_backend='gpu',
      max_episode_steps=200,
      control_mode=None,
      num_frames=3,
      seed=0,
      **kwargs,
  ):
    import os
    # Must happen before JAX initialises to avoid OOM when sharing one GPU.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.4')

    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: registers all tasks
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper

    self._num_envs  = num_envs
    self._obs_mode  = obs_mode
    self._num_frames = num_frames
    self._device    = 'cuda'

    # ------------------------------------------------------------------ #
    #  Build base GPU-vectorized env (identical to TD-MPC2 make_envs)
    # ------------------------------------------------------------------ #
    make_kwargs = dict(
        id=task,
        obs_mode=obs_mode,
        render_mode='rgb_array' if 'rgb' in obs_mode else None,
        sensor_configs=dict(width=image_size, height=image_size),
        num_envs=num_envs,
        sim_backend=sim_backend,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
    if control_mode is not None:
      make_kwargs['control_mode'] = control_mode

    env = gym.make(**make_kwargs)

    if 'rgb' in obs_mode:
      # Merges all camera RGB channels + exposes 'state' key for proprio
      env = FlattenRGBDObservationWrapper(
          env, rgb=True, depth=False, state=True)

    # ignore_terminations=True: episodes end only on truncation (fixed length)
    # record_metrics=True: populates info['final_info']['episode']
    self._env = ManiSkillVectorEnv(
        env, ignore_terminations=True, record_metrics=True)

    # Warm reset to discover obs/act shapes
    obs, _ = self._env.reset(seed=seed)
    self._setup_spaces(obs)

    if 'rgb' in obs_mode and num_frames > 1:
      self._setup_frame_stack(obs)

    # Track per-env done to compute is_first on the NEXT step
    self._prev_done = np.ones(num_envs, dtype=bool)

  # ------------------------------------------------------------------ #
  #  Shape discovery
  # ------------------------------------------------------------------ #

  def _setup_spaces(self, obs):
    self._act_dim   = self._env.action_space.shape[-1]
    state           = self._extract_state(obs)   # [N, state_dim] numpy
    self._state_dim = state.shape[-1]

    if 'rgb' in self._obs_mode:
      rgb = obs['rgb']                           # [N, H, W, C]
      self._img_h = rgb.shape[1]
      self._img_w = rgb.shape[2]
      self._img_c = rgb.shape[3]                 # channels per single frame
      # DreamerV3 CNN is channels-last: [H, W, C*num_frames]
      self._img_shape = (
          self._img_h,
          self._img_w,
          self._img_c * self._num_frames,
      )

  def _setup_frame_stack(self, obs):
    import torch
    rgb = obs['rgb'].float() / 255.0             # [N, H, W, C]
    N, H, W, C = rgb.shape
    # Circular buffer: [N, H, W, C, num_frames]
    self._frame_buf = (
        rgb.unsqueeze(-1)
           .expand(-1, -1, -1, -1, self._num_frames)
           .clone()
    )
    self._stack_ptr = 0

  # ------------------------------------------------------------------ #
  #  embodied.Env interface
  # ------------------------------------------------------------------ #

  @property
  def obs_space(self):
    spaces = {
        'reward':      elements.Space(np.float32, ()),
        'is_first':    elements.Space(bool, ()),
        'is_last':     elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
        'state':       elements.Space(np.float32, (self._state_dim,)),
    }
    if 'rgb' in self._obs_mode:
      # uint8 is required — DreamerV3 Encoder asserts dtype==uint8
      spaces['image'] = elements.Space(np.uint8, self._img_shape)
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(
            np.float32, (self._act_dim,), low=-1.0, high=1.0),
        'reset':  elements.Space(bool, ()),
    }

  def step(self, action):
    """
    action: dict with
      'reset'  [N] bool   — which envs to reset this step
      'action' [N, act_dim] float32

    Returns obs dict with all values shaped [N, ...] as numpy arrays.
    Driver._step_batched() slices axis-0 and fires one callback per env.
    """
    import torch

    reset_mask = np.asarray(action['reset'], dtype=bool)   # [N]

    # ---- Reset path -------------------------------------------------- #
    if reset_mask.any():
      reset_idx = torch.tensor(
          np.where(reset_mask)[0], device=self._device)
      obs, _ = self._env.reset(options={'env_idx': reset_idx})

      if 'rgb' in self._obs_mode:
        # Re-fill frame stack for reset envs with the new first frame
        rgb = obs['rgb'].float() / 255.0
        for _ in range(self._num_frames):
          self._frame_buf[reset_mask] = (
              rgb[reset_mask]
              .unsqueeze(-1)
              .expand(-1, -1, -1, -1, self._num_frames)
              .clone()
          )

      is_first   = reset_mask.copy()
      self._prev_done = np.zeros(self._num_envs, dtype=bool)

      return self._make_obs(
          obs,
          reward     = np.zeros(self._num_envs, dtype=np.float32),
          terminated = np.zeros(self._num_envs, dtype=bool),
          truncated  = np.zeros(self._num_envs, dtype=bool),
          is_first   = is_first,
      )

    # ---- Normal step ------------------------------------------------- #
    act = torch.tensor(
        np.asarray(action['action']),
        dtype=torch.float32, device=self._device)

    obs, reward, terminated, truncated, info = self._env.step(act)

    done = (terminated | truncated).cpu().numpy().astype(bool)   # [N]

    # Replace obs with final_observation for done envs so the world model
    # bootstraps from the true last state, not the post-reset obs.
    # Mirrors online_trainer.py lines in TD-MPC2 exactly.
    if done.any() and 'final_observation' in info:
      final = info['final_observation']
      import torch as _t
      done_t = _t.tensor(done, dtype=_t.bool, device=self._device)
      for k in obs:
        if k in final:
          expand = done_t.view(-1, *([1] * (obs[k].dim() - 1)))
          obs[k] = _t.where(expand.expand_as(obs[k]), final[k], obs[k])

    is_first        = self._prev_done.copy()
    self._prev_done = done.copy()

    return self._make_obs(
        obs,
        reward     = reward.cpu().numpy().astype(np.float32),
        terminated = terminated.cpu().numpy().astype(bool),
        truncated  = truncated.cpu().numpy().astype(bool),
        is_first   = is_first,
    )

  # ------------------------------------------------------------------ #
  #  Internal helpers
  # ------------------------------------------------------------------ #

  def _extract_state(self, obs):
    """
    Concatenate ManiSkill proprioceptive obs into a flat [N, D] float32 array.

    FlattenRGBDObservationWrapper (used in RGB mode) exposes obs['state'].
    State-only mode exposes obs['agent'] and obs['extra'] dicts.
    """
    import torch
    if 'state' in obs:
      return obs['state'].cpu().float().numpy()
    parts = []
    for v in obs.get('agent', {}).values():
      parts.append(v.reshape(v.shape[0], -1))
    for v in obs.get('extra', {}).values():
      parts.append(v.reshape(v.shape[0], -1))
    if not parts:
      raise ValueError(
          f'No state found in obs keys: {list(obs.keys())}. '
          f'Expected "state", "agent", or "extra".')
    return torch.cat(parts, dim=-1).cpu().float().numpy()

  def _stack_frames(self, obs):
    """
    Push latest RGB frame into circular buffer and return stacked image.

    Returns numpy uint8 [N, H, W, C*num_frames]  ← channels-LAST.

    When num_frames=1, skips the circular buffer entirely and returns the
    raw frame directly (no stacking overhead).

    DreamerV3's Encoder (rssm.py Encoder.__call__) does:
      x = jnp.concatenate(imgs, -1)   # concat on last axis → needs [H,W,C]
      B, H, W, C = x.shape            # unpacks channels-last
    So channels-last is required.  TD-MPC2 uses channels-first [N,C*F,H,W]
    because PyTorch Conv2d expects that — do NOT copy TD-MPC2 here.
    """
    import torch
    rgb = obs['rgb'].float() / 255.0              # [N, H, W, C]

    if self._num_frames == 1:
      return (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    self._frame_buf[..., self._stack_ptr] = rgb
    self._stack_ptr = (self._stack_ptr + 1) % self._num_frames

    # Roll so the oldest frame comes first in the stack
    stacked = self._frame_buf.roll(
        shifts=-self._stack_ptr, dims=-1)         # [N, H, W, C, F]
    N, H, W, C, F = stacked.shape
    # Merge C and F into channels-last: [N, H, W, C*F]
    stacked = stacked.reshape(N, H, W, C * F)
    return (stacked.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

  def _make_obs(self, obs, reward, terminated, truncated, is_first):
    done = terminated | truncated
    out = {
        'reward':      reward,
        'is_first':    is_first,
        'is_last':     done,
        'is_terminal': terminated,
        'state':       self._extract_state(obs),
    }
    if 'rgb' in self._obs_mode:
      out['image'] = self._stack_frames(obs)
    return out

  def close(self):
    self._env.close()