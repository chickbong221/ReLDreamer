import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  # ManiSkill episode-level metrics that TD-MPC2 logs once per completed
  # vector episode batch as mean over num_envs.
  #
  # These are emitted by ManiSkill only at episode end through final_info.
  # We bypass Dreamer's epstats aggregation for these keys so W&B receives
  # one point per completed vector batch, matching TD-MPC2.
  maniskill_metric_keys = (
      'log/success_once',
      'log/success_at_end',
      'log/fail_once',
  )
  maniskill_batch = {
      'count': 0,
      'values': collections.defaultdict(list),
  }

  def log_maniskill_vector_batch():
    values = maniskill_batch['values']
    if not values:
      return

    payload = {}
    for key, vals in values.items():
      if not vals:
        continue
      # Convert:
      #   log/success_once   -> train/success_once
      #   log/success_at_end -> train/success_at_end
      #   log/fail_once      -> train/fail_once
      payload[f'train/{key[len("log/"):]}'] = float(np.mean(vals))

    if payload:
      # Important:
      # Use Dreamer's logger instead of direct wandb.log().
      # This keeps all queued Dreamer summaries and the ManiSkill metrics
      # in one ordered logging stream, avoiding W&B step-order warnings.
      logger.add(payload)
      logger.write()

    values.clear()
    maniskill_batch['count'] = 0
  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')

      elif key.startswith('log/') and tran['is_last']:
        # For ManiSkill vectorized GPU envs, do not pass these metrics through
        # Dreamer's epstats aggregator. TD-MPC2 logs them immediately once per
        # completed vector episode batch, so we handle them separately below.
        if getattr(driver, 'batched', False) and key in maniskill_metric_keys:
          continue

        # Only record at episode end — these are episode-level metrics.
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key, value, agg='avg')

    # Match TD-MPC2 logging:
    #
    # TD-MPC2 receives ManiSkill final_info with tensors of shape [num_envs],
    # then logs:
    #
    #   metrics[k] = v.float().mean().item()
    #
    # Dreamer's batched driver calls callbacks once per sub-env transition, so
    # here we collect the final per-env scalar values and log their mean after
    # one full vector batch has completed.
    if getattr(driver, 'batched', False) and tran['is_last']:
      for key in maniskill_metric_keys:
        if key not in tran:
          continue
        value = np.asarray(tran[key])
        assert value.ndim == 0, (key, value.shape, value.dtype)
        maniskill_batch['values'][key].append(float(value))

      maniskill_batch['count'] += 1

      if maniskill_batch['count'] >= driver.length:
        log_maniskill_vector_batch()

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()