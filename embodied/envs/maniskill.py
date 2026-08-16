"""
ManiSkill GPU-vectorized environment wrapper for DreamerV3.

This wrapper mirrors the ManiSkill parts of TD-MPC2:
  - one instance wraps N parallel GPU envs via ManiSkillVectorEnv
  - training env and eval env can have different vector widths
  - eval env can be created with is_eval=True and reconfiguration_freq
  - final_info['episode'] metrics are exposed for TD-MPC2-style logging
  - eval render frames are available for videos/eval_video

Key difference from TD-MPC2 pixel wrapper:
  - TD-MPC2 returns channels-first [N, C*F, H, W] for PyTorch
  - DreamerV3 expects channels-last [N, H, W, C*F] for JAX
  - images are uint8; DreamerV3 normalizes internally
"""

import numpy as np
import embodied
import elements


def _holdout_set(value):
  """Canonical object keys split off for the few-shot phase.

  Comma-separated rather than a list because elements.Config rejects an empty
  list outright -- it cannot infer the element type of one -- and the common
  case is no holdout at all.
  """
  parts = value.split(',') if isinstance(value, str) else list(value or ())
  return {p.strip() for p in map(str, parts) if p.strip()}


def _flag(value, auto, name):
  """Parse an auto/true/false config value; 'auto' resolves to ``auto``.

  elements.Config hands command-line overrides through as strings, and bool()
  of a non-empty string is True, so 'false' has to be read rather than cast.
  """
  if isinstance(value, str):
    text = value.strip().lower()
    if text == 'auto':
      return bool(auto)
    if text in ('true', 'yes', '1'):
      return True
    if text in ('false', 'no', '0'):
      return False
    raise ValueError(f'{name} must be auto, true or false; got {value!r}')
  return bool(value)


def _use_named_cameras(setting, graph_cfg):
  """Whether to emit one image key per camera instead of one concatenated one.

  'auto' follows the graph so older configs are unchanged. Pinning it True is
  what lets a graph-free baseline see exactly the pixels the graph run sees,
  instead of silently switching to the concatenated single-image encoder.
  """
  return _flag(setting, graph_cfg.get('enabled', False), 'named_cameras')


def _plan_objects(plan):
  """Canonical object keys a task plan's subtasks name."""
  from scenegraph.core.affordance import canonical_affordance_key
  out = set()
  for entry in getattr(plan, 'subtasks', []) or []:
    obj_id = getattr(entry, 'obj_id', None)
    if obj_id:
      canonical = canonical_affordance_key(str(obj_id))
      if canonical:
        out.add(str(canonical))
  return out


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
      eval_reconfiguration_frequency=0,
      render_mode=None,
      mshab_task=None,
      mshab_split='train',
      mshab_eval_split=None,
      mshab_obj='all',
      mshab_num_build_configs=0,
      mshab_holdout_objs='',
      mshab_holdout_mode='auto',
      named_cameras='auto',
      instruction_table='',
      nonprivileged_obs=False,
      max_depth=20000.0,
      max_episode_steps=None,
      eval_max_episode_steps=None,
      frame_stack=1,
      graph=None,
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
    self._max_depth = float(max_depth)
    self._mshab_active = mshab_task is not None and mshab_task != 'none'
    self._graph_cfg = dict(graph or {})
    # One uint8 image per named camera, no depth. Mutually exclusive with the
    # concatenated single-image path below.
    #
    # Deliberately not tied to the graph: a graph-free baseline has to see the
    # same pixels this does, or turning the graph off would also swap the
    # visual encoder's input and the two runs would stop being comparable.
    # 'auto' follows the graph so existing configs are unchanged.
    self._named_cams = (
        self._mshab_active and 'rgb' in obs_mode
        and _use_named_cameras(named_cameras, self._graph_cfg))
    self._camera_keys = {}
    if self._named_cams:
      if self._num_frames != 1:
        raise ValueError(
            f'named cameras need num_frames=1, got {self._num_frames}; the '
            'RSSM supplies temporal context')
      cams = list(self._graph_cfg.get('cameras') or ['fetch_head', 'fetch_hand'])
      self._camera_keys = {
          'image_' + cam.rsplit('_', 1)[-1]: cam for cam in cams}
    self._rgb_obs = 'rgb' in obs_mode and not self._named_cams
    # MSHab depth path. 'depth+segmentation' takes the same wrappers as plain
    # 'depth'; the extra texture only feeds the scene graph.
    self._depth_obs = self._mshab_active and 'depth' in obs_mode
    # 0/None → keep the env's registered default; positive → override.
    # Eval gets its own horizon to mirror mshab/configs/sac_pick.yml
    # (train 100, eval 200) — falls back to max_episode_steps if unset.
    def _resolve_horizon(value):
      if value is None or int(value) <= 0:
        return None
      return int(value)
    train_override = _resolve_horizon(max_episode_steps)
    eval_override = _resolve_horizon(eval_max_episode_steps)
    if self._is_eval and eval_override is not None:
      self._max_episode_steps_override = eval_override
    else:
      self._max_episode_steps_override = train_override

    # TD-MPC2 uses cfg.render_mode, default rgb_array, and the eval env gets
    # video recording. For Dreamer, always make eval renderable; for training,
    # RGB observations also require rgb_array rendering/sensors.
    if render_mode is None:
      render_mode = 'rgb_array' if ('rgb' in obs_mode or is_eval) else None

    # ------------------------------------------------------------------ #
    #  Build base GPU-vectorized env, matching TD-MPC2 make_envs().
    # ------------------------------------------------------------------ #
    make_kwargs = dict(
        id=task,
        obs_mode=obs_mode,
        render_mode=render_mode,
        sensor_configs=dict(width=image_size, height=image_size),
        num_envs=self._num_envs,
        sim_backend=sim_backend,
        # Do not pass max_episode_steps. Use the registered default, same as
        # TD-MPC2, which reads it back with gym_utils.find_max_episode_steps_value.
        **kwargs,
    )
    if control_mode is not None:
      make_kwargs['control_mode'] = control_mode

    if self._mshab_active:
      import mshab.envs  # noqa: F401 registers mshab tasks
      from mani_skill import ASSET_DIR
      from mshab.envs.planner import plan_data_from_file
      subtask = task.split('SubtaskTrain')[0].lower()
      rearrange_dir = ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange'
      split = mshab_eval_split if (self._is_eval and mshab_eval_split) else mshab_split
      plan_data = plan_data_from_file(
          rearrange_dir / 'task_plans' / mshab_task / subtask / split / f'{mshab_obj}.json'
      )
      # A plan is one (build_config, init_config) pair, so there are many plans
      # per build config: keep whole build configs, matching sac/envs.py.
      task_plans = plan_data.plans
      # Few-shot split, filtered per plan so whole (build_config, init_config)
      # pairs stay intact the way the build-config filter below keeps them.
      # 'auto' pretrains on the seen categories and evaluates on the held-out
      # ones; the finetune env trains on 'only', which is why the mode is
      # separate from is_eval.
      holdout = _holdout_set(mshab_holdout_objs)
      if holdout:
        mode = str(mshab_holdout_mode or 'auto')
        if mode == 'auto':
          mode = 'only' if self._is_eval else 'exclude'
        if mode not in ('exclude', 'only'):
          raise ValueError(
              f'mshab_holdout_mode must be auto, exclude or only; got {mode!r}')
        want = (mode == 'only')
        seen = set().union(*(_plan_objects(p) for p in task_plans)) \
            if task_plans else set()
        # A key that names nothing removes nothing, and mode='exclude' would
        # then pretrain on the very categories the few-shot claim holds out.
        # Only 'only' fails on its own below, so check the keys themselves.
        unknown = holdout - seen
        if unknown:
          raise ValueError(
              f'mshab_holdout_objs {sorted(unknown)} match no object in split '
              f'{split!r} (it has {sorted(seen)}). Pass canonical keys as a '
              "comma-separated string, e.g. '024_bowl,013_apple' -- a list "
              'literal is read as one key and would silently hold out nothing.')
        task_plans = [
            p for p in task_plans if bool(_plan_objects(p) & holdout) == want]
        print(f'[env] holdout {sorted(holdout)} mode={mode}: '
              f'{len(task_plans)}/{len(plan_data.plans)} plans kept')
        if not task_plans:
          raise ValueError(
              f'holdout {sorted(holdout)} with mode={mode!r} left no plans in '
              f'split {split!r}; check the canonical object keys against the '
              'instruction table')
      n_bc = int(mshab_num_build_configs or 0)
      if not self._is_eval and n_bc > 0:
        all_names = sorted({p.build_config_name for p in task_plans})
        if n_bc < len(all_names):
          keep = set(all_names[:n_bc])
          task_plans = [p for p in task_plans if p.build_config_name in keep]
          print(f'[env] using {n_bc}/{len(all_names)} build configs '
                f'({len(task_plans)}/{len(plan_data.plans)} plans, train)')
      make_kwargs['task_plans'] = task_plans
      make_kwargs['scene_builder_cls'] = plan_data.dataset
      make_kwargs['spawn_data_fp'] = (
          rearrange_dir / 'spawn_data' / mshab_task / subtask / split / 'spawn_data.pt'
      )
      # mshab envs assert num_envs % num_scenes == 0 by default (63 train / 21 val).
      # Disable so any num_envs works (make_agent uses 1 for shape discovery).
      make_kwargs['require_build_configs_repeated_equally_across_envs'] = False
      # Use normalised rewards matching mshab's own training code.
      make_kwargs.setdefault('reward_mode', 'normalized_dense')
      # Match mshab/configs/sac_pick.yml: training horizon 100, eval 200.
      # is_eval selects eval_max_episode_steps above.
      if self._max_episode_steps_override is not None:
        make_kwargs['max_episode_steps'] = self._max_episode_steps_override
      # Match SAC's shader_dir to keep depth rendering identical.
      make_kwargs.setdefault('shader_dir', 'minimal')
    if is_eval:
      # TD-MPC2 creates a separate eval env and adds
      # reconfiguration_freq=cfg.eval_reconfiguration_frequency.
      make_kwargs['reconfiguration_freq'] = eval_reconfiguration_frequency

    env = gym.make(**make_kwargs)

    if nonprivileged_obs:
      from embodied.envs.obs_wrappers import NonPrivilegedObsWrapper
      env = NonPrivilegedObsWrapper(env)

    if self._rgb_obs:
      # Matches TD-MPC2's RGB path: flatten RGBD observation into rgb + state.
      env = FlattenRGBDObservationWrapper(
          env, rgb=True, depth=False, state=True)

    self._named_wrapper = None
    if self._named_cams:
      from embodied.envs.obs_wrappers import NamedCameraRGBWrapper
      env = NamedCameraRGBWrapper(env, self._camera_keys)
      self._named_wrapper = env

    # MSHab depth path: apply the same per-env wrappers as
    # mshab/mshab/envs/make.py (minus FrameStack, since DreamerV3's RSSM
    # supplies temporal context). FetchDepthObservationWrapper produces
    # {state, fetch_head_depth, fetch_hand_depth} with the depth tensors
    # permuted to channel-first.
    if self._depth_obs:
      from mshab.envs.wrappers import FetchDepthObservationWrapper
      env = FetchDepthObservationWrapper(
          env, cat_state=True, cat_pixels=False)

    # stationary_head=True matches sac_pick.yml. Applies to every MS-HAB
    # observation mode, not just depth.
    if self._mshab_active:
      from mshab.envs.wrappers import FetchActionWrapper
      env = FetchActionWrapper(
          env,
          stationary_base=False,
          stationary_torso=False,
          stationary_head=True)

    # Read the registered horizon before vector wrapping, same as TD-MPC2.
    from mani_skill.utils import gym_utils as _gym_utils
    self._max_episode_steps = _gym_utils.find_max_episode_steps_value(env)

    # Keep the pre-vector wrapper env for eval rendering. TD-MPC2's
    # RecordEpisodeWrapper also records from the eval env before the final
    # ManiSkillVectorEnv wrapper.
    self._render_env = env

    # ignore_terminations=True: episodes end only on truncation.
    # record_metrics=True: populates info['final_info']['episode'].
    # The horizon reaches the env through make_kwargs above; ManiSkillVectorEnv
    # forwards **kwargs to gym.make only when env is a string, so passing it
    # here would be silently dropped.
    self._env = ManiSkillVectorEnv(
        env, ignore_terminations=True, record_metrics=True)

    from scenegraph.adapters.graph_obs import build_graph_obs
    self._graph = build_graph_obs(
        self._env, self._graph_cfg, num_envs=self._num_envs,
        sensor_source=self._named_wrapper)

    # Shared across every method being compared: the same table, the same
    # lookup, so only what each model does with the vector differs.
    self._instr = None
    if instruction_table:
      from embodied.envs.instruction import InstructionReader, InstructionTable
      self._instr = InstructionReader(
          self._env, InstructionTable(instruction_table), self._num_envs)

    # Warm reset to discover obs/act shapes.
    obs, _ = self._env.reset(seed=seed)
    obs = self._obs_to_dict(obs)
    self._setup_spaces(obs)
    self._graph_obs = (
        self._graph.reset() if self._graph is not None else {})
    self._instr_obs = (
        {'instruction': self._instr.step()} if self._instr is not None else {})

    if self._rgb_obs and self._num_frames > 1:
      self._setup_frame_stack(obs)

    # Track per-env done to compute is_first on the next step.
    self._prev_done = np.ones(self._num_envs, dtype=bool)

    # Full ManiSkill final_info['episode'] from the most recent completed vector
    # episode. train.py eval uses this to mirror TD-MPC2 final_info_metrics().
    self._last_episode_metrics = {}

  @property
  def is_batched(self):
    return True

  @property
  def num_envs(self):
    return self._num_envs

  @property
  def graph_vocab_sizes(self):
    """Whitelist-dependent vocabulary sizes the agent cannot derive itself."""
    if self._graph is None:
      return {}
    return {
        'entity_vocab': self._graph.vocab.sizes['entity'],
        'uid_vocab': self._graph.uid_vocab,
    }

  # ------------------------------------------------------------------ #
  #  Shape discovery
  # ------------------------------------------------------------------ #

  def _setup_spaces(self, obs):
    self._act_dim = self._env.action_space.shape[-1]
    state = self._extract_state(obs)
    self._state_dim = state.shape[-1]

    if self._rgb_obs:
      rgb = obs['rgb']
      self._img_h = rgb.shape[1]
      self._img_w = rgb.shape[2]
      self._img_c = rgb.shape[3]
      self._img_shape = (
          self._img_h,
          self._img_w,
          self._img_c * self._num_frames,
      )

    if self._named_cams:
      shapes = {tuple(obs[k].shape[1:]) for k in self._camera_keys}
      if len(shapes) != 1:
        raise ValueError(f'named cameras disagree on shape: {shapes}')
      self._named_shape = shapes.pop()

    if self._depth_obs:
      # FetchDepthObservationWrapper permutes to [N, 1, H, W].
      d = obs['fetch_head_depth']
      self._depth_h = d.shape[-2]
      self._depth_w = d.shape[-1]
      self._depth_shape = (self._depth_h, self._depth_w, 1)

  def _setup_frame_stack(self, obs):
    rgb = obs['rgb'].float() / 255.0  # [N, H, W, C]
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
        # log/ prefix: excluded from the world model by the notlog filter in
        # make_agent. train.py reads these only at is_last=True.
        'log/success_once': elements.Space(np.float32, ()),
        'log/success_at_end': elements.Space(np.float32, ()),
        'log/fail_once': elements.Space(np.float32, ()),
    }
    if self._rgb_obs:
      spaces['image'] = elements.Space(np.uint8, self._img_shape)
    for key in self._camera_keys:
      spaces[key] = elements.Space(np.uint8, self._named_shape)
    if self._graph is not None:
      # Registry overflow means a retained vertex displaced a current one.
      # log/ keeps it out of the world model; train.py reads it at is_last.
      spaces['log/graph_overflow_drops'] = elements.Space(np.float32, ())
      # Facts truncated at e_max. Nonzero means spatial edges are being lost;
      # graph_pack keeps physical, then affordance, then spatial.
      spaces['log/graph_fact_drops'] = elements.Space(np.float32, ())
      # Fraction of frames whose graph names no target vertex. Anything but
      # near-zero means the goal flag is dark and the semantic target loss is
      # training on nothing.
      spaces['log/graph_target_missing'] = elements.Space(np.float32, ())
      # Entries in the caches that outlive an episode. Flat is healthy; a
      # steady climb over millions of steps is the leak signature.
      spaces['log/graph_cache_entries'] = elements.Space(np.float32, ())
    if self._depth_obs:
      # Store raw millimetres as uint16 to match MSHab's SAC replay buffer
      # (mshab/mshab/agents/sac/replay.py). This is 4x smaller than float32
      # and 2x larger than uint8 but lossless at 1 mm precision.
      hi = int(self._max_depth)
      spaces['depth_head'] = elements.Space(
          np.uint16, self._depth_shape, 0, hi)
      spaces['depth_hand'] = elements.Space(
          np.uint16, self._depth_shape, 0, hi)
    if self._graph is not None:
      dtypes = self._graph.obs_spec_dtypes
      for key, shape in self._graph.obs_spec_shapes.items():
        spaces[key] = elements.Space(dtypes[key], shape)
    if self._instr is not None:
      spaces['instruction'] = elements.Space(
          np.float32, (self._instr.table.dim,))
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

    Returns obs dict with all values shaped [N, ...]. Driver._step_batched()
    slices axis 0 and fires one callback per sub-env.
    """
    import torch

    reset_mask = np.asarray(action['reset'], dtype=bool)  # [N]

    # ---- Reset path -------------------------------------------------- #
    if reset_mask.any():
      reset_idx = torch.tensor(np.where(reset_mask)[0], device=self._device)
      obs, _ = self._env.reset(options={'env_idx': reset_idx})
      obs = self._obs_to_dict(obs)

      if self._rgb_obs and self._num_frames > 1:
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
      if self._graph is not None:
        self._graph_obs = self._graph.step(is_first=is_first)
      if self._instr is not None:
        self._instr_obs = {'instruction': self._instr.step()}

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
        np.asarray(action['action']), dtype=torch.float32, device=self._device)

    obs, reward, terminated, truncated, info = self._env.step(act)
    obs = self._obs_to_dict(obs)
    done = (terminated | truncated).cpu().numpy().astype(bool)  # [N]

    # Replace obs with final_observation for done envs so the replay/eval code
    # sees the true last state, mirroring TD-MPC2 online_trainer.py.
    if done.any() and 'final_observation' in info:
      final = self._obs_to_dict(info['final_observation'])
      done_t = torch.tensor(done, dtype=torch.bool, device=self._device)
      for key in obs:
        if key in final and isinstance(obs[key], torch.Tensor):
          expand = done_t.view(-1, *([1] * (obs[key].dim() - 1)))
          obs[key] = torch.where(expand.expand_as(obs[key]), final[key], obs[key])

    # ---- Extract final_info['episode'] metrics ----------------------- #
    log_metrics = {}
    if done.any() and 'final_info' in info:
      ep_info = info['final_info'].get('episode', {})
      self._last_episode_metrics = {}
      for key, vals in ep_info.items():
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
    if self._graph is not None:
      # The vector env auto-resets inside step(), so a done env's sensors
      # already belong to the next episode; those frames re-emit the previous
      # graph rather than one built from the wrong episode.
      self._graph_obs = self._graph.step(is_first=is_first, is_last=done)
    if self._instr is not None:
      # Same reason as the graph: a done env's subtask pointer has already
      # moved to the next episode, so that frame re-emits its previous row.
      self._instr_obs = {'instruction': self._instr.step(is_last=done)}

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

  def _obs_to_dict(self, obs):
    # obs_mode='state' makes ManiSkill return a flat Tensor instead of a dict.
    # Wrap it so the rest of this class can treat obs uniformly as a mapping.
    import torch
    if isinstance(obs, torch.Tensor):
      return {'state': obs}
    return obs

  def _extract_state(self, obs):
    """Concatenate ManiSkill proprioceptive obs into [N, D] float32."""
    import torch
    if 'state' in obs:
      return obs['state'].cpu().float().numpy()
    parts = []
    for value in obs.get('agent', {}).values():
      parts.append(value.reshape(value.shape[0], -1))
    for value in obs.get('extra', {}).values():
      parts.append(value.reshape(value.shape[0], -1))
    if not parts:
      raise ValueError(
          f'No state found in obs keys: {list(obs.keys())}. '
          f'Expected "state", "agent", or "extra".')
    return torch.cat(parts, dim=-1).cpu().float().numpy()

  def _extract_depth(self, obs):
    # Depth from ManiSkill is int16 millimetres in [N, 1, H, W]. Mirror MSHab's
    # SAC pipeline: keep raw mm and store as uint16 (clamped to [0, max_depth]).
    # Normalisation to [0, 1] happens inside the encoder/decoder, not here.
    import torch
    hi = int(self._max_depth)
    def _to_uint16(t):
      t = t.permute(0, 2, 3, 1).contiguous()
      t = torch.clamp(t, 0, hi)
      return t.to(torch.int32).cpu().numpy().astype(np.uint16)
    return _to_uint16(obs['fetch_head_depth']), _to_uint16(obs['fetch_hand_depth'])

  def _stack_frames(self, obs):
    """Return uint8 image [N, H, W, C*num_frames] in channels-last layout."""
    rgb = obs['rgb'].float() / 255.0  # [N, H, W, C]

    if self._num_frames == 1:
      return (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    self._frame_buf[..., self._stack_ptr] = rgb
    self._stack_ptr = (self._stack_ptr + 1) % self._num_frames

    stacked = self._frame_buf.roll(shifts=-self._stack_ptr, dims=-1)
    num_envs, height, width, channels, frames = stacked.shape
    stacked = stacked.reshape(num_envs, height, width, channels * frames)
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
    if self._rgb_obs:
      out['image'] = self._stack_frames(obs)
    for key in self._camera_keys:
      out[key] = obs[key].cpu().numpy().astype(np.uint8)
    if self._graph is not None:
      out['log/graph_overflow_drops'] = self._graph.overflow_drops
      out['log/graph_fact_drops'] = self._graph.fact_drops
      out['log/graph_target_missing'] = self._graph.target_missing
      out['log/graph_cache_entries'] = np.full(
          self._num_envs, float(self._graph.cache_entries), np.float32)
    if self._depth_obs:
      depth_head, depth_hand = self._extract_depth(obs)
      out['depth_head'] = depth_head
      out['depth_hand'] = depth_hand
    out.update(self._graph_obs)
    out.update(self._instr_obs)
    if log_metrics:
      out.update(log_metrics)
    return out

  def last_episode_metrics(self):
    return {
        key: np.array(value, copy=True)
        for key, value in self._last_episode_metrics.items()}

  def render(self):
    """Return eval RGB frames as [num_envs, H, W, 3] for W&B video."""
    from mani_skill.utils import common

    candidates = []
    for candidate in (getattr(self, '_render_env', None), self._env):
      if candidate is not None and candidate not in candidates:
        candidates.append(candidate)
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
