import importlib
import os
import pathlib
import sys
from functools import partial as bind

# Must be set before JAX initialises to share GPU with ManiSkill cleanly.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.4')

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  suite = config.task.split('_', 1)[0]

  # ------------------------------------------------------------------ #
  #  ManiSkill: intercept Driver construction inside run.train.
  #
  #  train.py does:
  #    fns = [bind(make_env, i) for i in range(args.envs)]
  #    driver = embodied.Driver(fns, parallel=not args.debug)
  #
  #  We set args.envs=1 so fns has exactly one element.
  #  We patch embodied.Driver so that when train.py calls Driver(fns, ...),
  #  it gets our batched Driver instead: fns[0]() builds the ManiSkill env
  #  and _ManiSkillDriver passes it to the batched init path.
  #  The patch is restored in a finally block so nothing else is affected.
  # ------------------------------------------------------------------ #
  if suite == 'maniskill':
    num_envs = dict(config.env.get('maniskill', {})).get('num_envs', 32)
    args = args.update(envs=1, debug=False)  # one env call, no subprocess

    _OrigDriver = embodied.Driver

    class _ManiSkillDriver(_OrigDriver):
      def __init__(self, make_env_fns, parallel=True, **kwargs):
        # fns[0]() returns the ManiSkill batched env instance
        env = make_env_fns[0]()
        _OrigDriver.__init__(
            self, env,
            batched=True,
            num_envs=num_envs,
            **kwargs,
        )

    embodied.Driver = _ManiSkillDriver
    try:
      if config.script == 'train':
        embodied.run.train(
            bind(make_agent, config),
            bind(make_replay, config, 'replay'),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args)
      else:
        raise NotImplementedError(
            f'ManiSkill only supports script=train, got {config.script}')
    finally:
      embodied.Driver = _OrigDriver
    return

  # ------------------------------------------------------------------ #
  #  All other envs — original code unchanged
  # ------------------------------------------------------------------ #
  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  suite = config.task.split('_', 1)[0]
  # For ManiSkill: build with num_envs=1 just for obs/act space discovery.
  # This avoids spinning up 32 GPU envs twice.
  if suite == 'maniskill':
    env = make_env(config, 0, num_envs=1)
  else:
    env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []

  # ── ADDED: extend filter to include epstats (success_once, fail_once, return)
  # and fps metrics so they reach wandb, in addition to the original filter.
  suite = config.task.split('_', 1)[0]
  log_filter = config.logger.filter
  if suite == 'maniskill':
    log_filter = log_filter + '|episode/score|epstats/log/|fps/'

  outputs.append(elements.logger.TerminalOutput(log_filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      # ── CHANGED: init wandb with full args matching TD-MPC2 convention,
      # then wrap WandBOutput to remap key names for comparability.
      import wandb as _wandb
      lc = config.logger
      _wandb.init(
          project=lc.get('wandb_project', 'dreamerv3'),
          entity=lc.get('wandb_entity') or None,
          name=lc.get('wandb_name') or '/'.join(logdir.split('/')[-4:]),
          group=lc.get('wandb_group') or None,
          dir=logdir,
          config=dict(config),
          resume='allow',
      )
      if suite == 'maniskill':
        # Wrap WandBOutput to remap keys to TD-MPC2 names on write.
        # DreamerV3's own keys are untouched; we only add aliases.
        base_wandb = elements.logger.WandBOutput(
            '/'.join(logdir.split('/')[-4:]))
        outputs.append(_ManiSkillWandBOutput(base_wandb, step))
      else:
        outputs.append(elements.logger.WandBOutput(
            '/'.join(logdir.split('/')[-4:])))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


# ── ADDED: key-remapping wrapper placed just before make_logger in main.py ──
# Maps DreamerV3 aggregated episode keys → TD-MPC2 wandb key names so both
# runs are directly comparable. Does NOT modify DreamerV3's own logged keys.
class _ManiSkillWandBOutput:

  _REMAP = {
      'epstats/log/success_once/avg': 'train/success_once',
      'epstats/log/fail_once/avg':    'train/fail_once',
      'episode/score':                'train/return',
      'fps/policy':                   'time/rollout_fps',
  }

  def __init__(self, base_output, step):
    self._base = base_output
    self._step = step

  def __call__(self, summaries):
    # ── ADDED: strip policy_ image summaries before passing to WandBOutput.
    # Images from logfn (epstats/policy_{key}) are stacked uint8 frames that
    # would be logged as video. We only want video from report (openloop/).
    filtered = [
        (s, k, v) for s, k, v in summaries
        if 'policy_' not in k          # blocks epstats/policy_{key} videos
    ]
    self._base(filtered)               # DreamerV3 keys written unchanged
    # Write TD-MPC2-named aliases for scalar metrics.
    try:
      import wandb as _wandb
      if _wandb.run is None:
        return
      aliases = {}
      for step_val, key, value in filtered:
        if key in self._REMAP and isinstance(value, (int, float, np.floating,
                                                       np.integer)):
          aliases[self._REMAP[key]] = float(value)
      if aliases:
        _wandb.log(aliases, step=int(self._step))
    except Exception:
      pass


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy':     'embodied.envs.dummy:Dummy',
      'gym':       'embodied.envs.from_gym:FromGym',
      'dm':        'embodied.envs.from_dmenv:FromDM',
      'crafter':   'embodied.envs.crafter:Crafter',
      'dmc':       'embodied.envs.dmc:DMC',
      'atari':     'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab':     'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav':   'embodied.envs.loconav:LocoNav',
      'pinpad':    'embodied.envs.pinpad:PinPad',
      'langroom':  'embodied.envs.langroom:LangRoom',
      'procgen':   'embodied.envs.procgen:ProcGen',
      'bsuite':    'embodied.envs.bsuite:BSuite',
      'memmaze':   lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
      # ManiSkill: builds the batched GPU env directly.
      # Skips wrap_env — batched obs are not compatible with scalar wrappers.
      'maniskill': 'embodied.envs.maniskill:ManiSkill',
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = dict(config.env.get(suite, {}))   # mutable copy
  kwargs.update(overrides)                   # e.g. num_envs=1 for space discovery
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  if suite == 'maniskill':
    return env   # no wrappers: Driver handles batched obs internally
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)
  return stream


if __name__ == '__main__':
  main()