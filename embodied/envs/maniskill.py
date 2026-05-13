"""
ManiSkill GPU-vectorized environment wrapper for DreamerV3.

Design mirrors TD-MPC2's envs/maniskill.py + envs/wrappers/pixels.py:
  - One instance wraps N parallel GPU envs via ManiSkillVectorEnv
  - Returns batched numpy dicts [N, ...] from step()
  - Driver(batched=True) splits these into N per-worker replay transitions
  - Frame stacking for RGB mode lives entirely on GPU

Key difference from TD-MPC2 pixel wrapper:
  - TD-MPC2 returns channels-first [N, C*F, H, W] (PyTorch convention)
  - DreamerV3 Encoder expects channels-last [H, W, C*F] (JAX/TF convention)
  - Images must be uint8 (Encoder asserts this and normalises internally)
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
      control_mode=None,
      num_frames=3,
      seed=0,
      is_eval=False,
      eval_reconfiguration_frequency=1,
      **kwargs,
  ):
    import gymnasium as gym
    import torch  # noqa: F401
    import mani_skill.envs  # noqa: F401 registers all tasks
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper

    self._num_envs = int(num_envs)
    self._obs_mode = obs_mode
    self._num_frames = int(num_frames)
    self._device = 'cuda'
    self._is_eval = bool(is_eval)

    # ------------------------------------------------------------------ #
    #  Build base GPU-vectorized env, matching TD-MPC2 make_envs.
    # ------------------------------------------------------------------ #
    make_kwargs = dict(
        id=task,
        obs_mode=obs_mode,
        # TD-MPC2 uses render_mode='rgb_array' for eval videos. Keep it
        # enabled for RGB training and for all eval envs, including state obs.
        render_mode='rgb_array' if ('rgb' in obs_mode or is_eval) else None,
        sensor_configs=dict(width=image_size, height=image_size),
        num_envs=self._num_envs,
        sim_backend=sim_backend,
        # Do NOT pass max_episode_steps. Use the environment's registered
        # default, same as TD-MPC2.
        **kwargs,
    )
    if control_mode is not None:
      make_kwargs['control_mode'] = control_mode
    if is_eval:
      # TD-MPC2 creates a separate eval env and passes
      # reconfiguration_freq=cfg.eval_reconfiguration_frequency.
      make_kwargs['reconfiguration_freq'] = eval_reconfiguration_frequency

    env = gym.make(**make_kwargs)

    from embodied.envs.obs_wrappers import NonPrivilegedObsWrapper
    env = NonPrivilegedObsWrapper(env)

    if 'rgb' in obs_mode:
      # Merges all camera RGB channels + exposes 'state' key for proprio.
      env = FlattenRGBDObservationWrapper(
          env, rgb=True, depth=False, state=True)

    # Read the registered horizon, same as TD-MPC2.
    from mani_skill.utils import gym_utils as _gym_utils
    self._max_episode_steps = _gym_utils.find_max_episode_steps_value(env)

    # ignore_terminations=True: episodes end only on truncation.
    # record_metrics=True: populates info['final_info']['episode'].
    self._env = ManiSkillVectorEnv(
        env, ignore_terminations=True, record_metrics=True)

    # Warm reset to discover obs/act shapes.
    obs, _ = self._env.reset(seed=seed)
    self._setup_spaces(obs)

    if 'rgb' in obs_mode and self._num_frames > 1:
      self._setup_frame_stack(obs)

    # Track per-env done to compute is_first on the NEXT step.
    self._prev_done = np.ones(self._num_envs, dtype=bool)

    # Full ManiSkill final_info['episode'] from the most recent completed
    # vector episode. train.py eval uses this to mirror TD-MPC2's
    # final_info_metrics() exactly without adding these values to replay.
    self._last_episode_metrics = {}

  @property
  def is_batched(self):
    return True

  @property
  def num_envs(self):
    return self._num_envs

  # ------------------------------------------------------------------ #
  #  Shape discovery
  # ------------------------------------------------------------------ #

  def _setup_spaces(self, obs):
    self._act_dim = self._env.action_space.shape[-1]
    state = self._extract_state(obs)   # [N, state_dim] numpy
    self._state_dim = state.shape[-1]

    if 'rgb' in self._obs_mode:
      rgb = obs['rgb']                 # [N, H, W, C]
      self._img_h = rgb.shape[1]
      self._img_w = rgb.shape[2]
      self._img_c = rgb.shape[3]
      # DreamerV3 CNN is channels-last: [H, W, C*num_frames]
      self._img_shape = (
          self._img_h,
          self._img_w,
          self._img_c * self._num_frames,
      )

  def _setup_frame_stack(self, obs):
    rgb = obs['rgb'].float() / 255.0   # [N, H, W, C]
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
        'reward': elements.Space(np.float32, ()),
        'is_first': elements.Space(bool, ()),
        'is_last': elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
        'state': elements.Space(np.float32, (self._state_dim,)),
        # log/ prefix: excluded from world model by the notlog filter in
        # make_agent. train.py reads these only at is_last=True.
        'log/success_once': elements.Space(np.float32, ()),
        'log/success_at_end': elements.Space(np.float32, ()),
        'log/fail_once': elements.Space(np.float32, ()),
    }
    if 'rgb' in self._obs_mode:
      # uint8 is required; DreamerV3 Encoder normalizes internally.
      spaces['image'] = elements.Space(np.uint8, self._img_shape)
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(
            np.float32, (self._act_dim,), low=-1.0, high=1.0),
        'reset': elements.Space(bool, ()),
    }

  def step(self, action):
    """
    action: dict with
      'reset'  [N] bool
      'action' [N, act_dim] float32

    Returns obs dict with all values shaped [N, ...] as numpy arrays.
    Driver._step_batched() slices axis 0 and fires one callback per env.
    """
    import torch

    reset_mask = np.asarray(action['reset'], dtype=bool)  # [N]

    # ---- Reset path -------------------------------------------------- #
    if reset_mask.any():
      reset_idx = torch.tensor(np.where(reset_mask)[0], device=self._device)
      obs, _ = self._env.reset(options={'env_idx': reset_idx})

      if 'rgb' in self._obs_mode and self._num_frames > 1:
        rgb = obs['rgb'].float() / 255.0
        self._frame_buf[reset_idx] = (
            rgb[reset_idx]
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, self._num_frames)
            .clone()
        )

      is_first = reset_mask.copy()
      self._prev_done[reset_mask] = False
      self._last_episode_metrics = {}

      return self._make_obs(
          obs,
          reward=np.zeros(self._num_envs, dtype=np.float32),
          terminated=np.zeros(self._num_envs, dtype=bool),
          truncated=np.zeros(self._num_envs, dtype=bool),
          is_first=is_first,
          log_metrics={},
      )

    # ---- Normal step ------------------------------------------------- #
    act = torch.tensor(
        np.asarray(action['action']),
        dtype=torch.float32, device=self._device)

    obs, reward, terminated, truncated, info = self._env.step(act)
    done = (terminated | truncated).cpu().numpy().astype(bool)  # [N]

    # Replace obs with final_observation for done envs so the world model
    # bootstraps from the true last state, not the post-reset obs. This mirrors
    # TD-MPC2's online_trainer.py handling.
    if done.any() and 'final_observation' in info:
      final = info['final_observation']
      done_t = torch.tensor(done, dtype=torch.bool, device=self._device)
      for k in obs:
        if k in final:
          expand = done_t.view(-1, *([1] * (obs[k].dim() - 1)))
          obs[k] = torch.where(expand.expand_as(obs[k]), final[k], obs[k])

    # ---- Extract final_info['episode'] metrics ----------------------- #
    log_metrics = {}
    if done.any() and 'final_info' in info:
      ep_info = info['final_info'].get('episode', {})
      self._last_episode_metrics = {}
      for key, vals in ep_info.items():
        # Store complete metric set that TD-MPC2 averages in
        # final_info_metrics(): return, episode_len, reward, success_once,
        # success_at_end, fail_once, fail_at_end when present.
        try:
          arr = vals.cpu().float().numpy().astype(np.float32)
        except AttributeError:
          arr = np.asarray(vals, dtype=np.float32)
        self._last_episode_metrics[key] = arr

      for key in ('success_once', 'success_at_end', 'fail_once'):
        if key in ep_info:
          vals = ep_info[key]
          arr = vals.cpu().float().numpy().astype(np.float32)
          arr = np.where(done, arr, 0.0)
          log_metrics[f'log/{key}'] = arr

    is_first = self._prev_done.copy()
    self._prev_done = done.copy()

    return self._make_obs(
        obs,
        reward=reward.cpu().numpy().astype(np.float32),
        terminated=terminated.cpu().numpy().astype(bool),
        truncated=truncated.cpu().numpy().astype(bool),
        is_first=is_first,
        log_metrics=log_metrics,
    )

  # ------------------------------------------------------------------ #
  #  Internal helpers
  # ------------------------------------------------------------------ #

  def _extract_state(self, obs):
    """Concatenate ManiSkill proprioceptive obs into [N, D] float32."""
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

    Returns numpy uint8 [N, H, W, C*num_frames] using channels-last layout.
    """
    rgb = obs['rgb'].float() / 255.0  # [N, H, W, C]

    if self._num_frames == 1:
      return (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    self._frame_buf[..., self._stack_ptr] = rgb
    self._stack_ptr = (self._stack_ptr + 1) % self._num_frames

    stacked = self._frame_buf.roll(
        shifts=-self._stack_ptr, dims=-1)  # [N, H, W, C, F]
    N, H, W, C, F = stacked.shape
    stacked = stacked.reshape(N, H, W, C * F)
    return (stacked.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

  def _make_obs(self, obs, reward, terminated, truncated, is_first,
                log_metrics=None):
    done = terminated | truncated
    out = {
        'reward': reward,
        'is_first': is_first,
        'is_last': done,
        'is_terminal': terminated,
        'state': self._extract_state(obs),
        # Always emit log/ keys so the replay schema is consistent. Non-done
        # steps get zeros; train.py only reads them at is_last=True.
        'log/success_once': np.zeros(self._num_envs, dtype=np.float32),
        'log/success_at_end': np.zeros(self._num_envs, dtype=np.float32),
        'log/fail_once': np.zeros(self._num_envs, dtype=np.float32),
    }
    if 'rgb' in self._obs_mode:
      out['image'] = self._stack_frames(obs)
    if log_metrics:
      out.update(log_metrics)
    return out

  def last_episode_metrics(self):
    return {
        k: np.array(v, copy=True)
        for k, v in self._last_episode_metrics.items()}

  def render(self):
    """Return eval RGB frames as [num_envs, H, W, 3] for W&B video."""
    from mani_skill.utils import common

    candidates = [self._env]
    for name in ('env', 'base_env', 'unwrapped'):
      try:
        candidate = getattr(self._env, name)
      except Exception:
        candidate = None
      if candidate is not None and candidate not in candidates:
        candidates.append(candidate)

    last_error = None
    for candidate in candidates:
      if not hasattr(candidate, 'render'):
        continue
      try:
        img = candidate.render()
        img = common.to_numpy(img)
        if img.ndim == 3:
          img = img[None]
        return img
      except Exception as exc:
        last_error = exc
    if last_error is not None:
      raise last_error
    raise AttributeError('No render() method found on ManiSkill env.')

  def close(self):
    self._env.close()
