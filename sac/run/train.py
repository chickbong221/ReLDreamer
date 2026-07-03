"""SAC training loop.

Single-process driver over a GPU-vectorized ManiSkill/MS-HAB env. One
``training_freq`` env-step block is interleaved with ``training_freq * utd``
gradient updates, then logging/eval/save checkpoints fire at their own
cadences.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import torch

from ..agent import SACAgent
from ..envs import action_box, adapt_obs, build_env
from ..graph_env import build_graph_obs
from ..replay import ReplayBuffer, build_obs_spec


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_replay(sample_obs, action_shape, num_envs, cfg, device):
    spec = build_obs_spec(sample_obs)
    return ReplayBuffer(
        obs_spec=spec,
        action_shape=action_shape,
        num_envs=num_envs,
        buffer_size=int(cfg["agent"]["buffer_size"]),
        sample_device=device,
    )


def _format_console_metrics(metrics: dict) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            parts.append(f"{key}: {int(value)}")
        elif isinstance(value, (float, np.floating)):
            parts.append(f"{key}: {float(value):.4g}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, logdir: str, outputs):
        self.logdir = logdir
        self.outputs = list(outputs)
        os.makedirs(logdir, exist_ok=True)
        self.tb = None
        self.jsonl_path = None
        if "tensorboard" in self.outputs:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(logdir)
        if "jsonl" in self.outputs:
            self.jsonl_path = os.path.join(logdir, "metrics.jsonl")
        self.wandb = None
        if "wandb" in self.outputs:
            import wandb
            self.wandb = wandb

    def scalar(self, tag: str, value, step: int) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        else:
            value = float(value)
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)
        if self.wandb is not None:
            self.wandb.log({tag: value}, step=step)
        if self.jsonl_path is not None:
            import json
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps({"step": step, "tag": tag, "value": value}) + "\n")

    def close(self):
        if self.tb is not None:
            self.tb.close()


# --------------------------------------------------------------------------- #
# Train entrypoint
# --------------------------------------------------------------------------- #
def train(config: dict) -> None:
    suite, task = config["task"].split("_", 1)
    if suite != "maniskill":
        raise NotImplementedError(
            f"sac.run.train only supports the 'maniskill' suite (got {suite!r}). "
            "MS-HAB envs are reached via task='maniskill_<SubtaskTrain>'.")
    env_cfg = dict(config["env"]["maniskill"])
    agent_cfg = dict(config["agent"])
    run_cfg = dict(config["run"])
    graph_cfg = dict(config.get("graph", {}))
    obs_mode = str(env_cfg["obs_mode"])
    graph_enabled = bool(graph_cfg.get("enabled", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(int(config["seed"]))

    # --- env ----------------------------------------------------------------
    envs = build_env(task, env_cfg, is_eval=False, seed=int(config["seed"]),
                     graph_enabled=graph_enabled)
    eval_envs = build_env(task, env_cfg, is_eval=True, seed=int(config["seed"]) + 1,
                          graph_enabled=graph_enabled)
    action_shape, action_low, action_high = action_box(envs)
    num_envs = int(env_cfg["num_envs"])
    num_eval_envs = int(env_cfg["num_eval_envs"])

    graph_train = build_graph_obs(envs, graph_cfg, num_envs=num_envs)
    graph_eval = build_graph_obs(eval_envs, graph_cfg, num_envs=num_eval_envs)

    # --- initial obs (also serves as sample for spec discovery) -------------
    obs_raw, _ = envs.reset(seed=int(config["seed"]))
    obs = adapt_obs(obs_raw, obs_mode, device)
    if graph_train is not None:
        obs.update(graph_train.reset(device))
    eval_obs_raw, _ = eval_envs.reset(seed=int(config["seed"]) + 1)
    eval_obs = adapt_obs(eval_obs_raw, obs_mode, device)
    if graph_eval is not None:
        eval_obs.update(graph_eval.reset(device))

    # --- agent + replay -----------------------------------------------------
    if graph_train is not None:
        agent_cfg.setdefault("encoder", {})
        agent_cfg["encoder"]["graph"] = {
            "node_vocab_size": len(graph_train.node_vocab),
            "edge_vocab_size": len(graph_train.edge_vocab),
            "embed_dim": graph_cfg.get("embed_dim", 64),
            "hidden_dim": graph_cfg.get("hidden_dim", 256),
            "out_dim": graph_cfg.get("out_dim", 128),
            "num_layers": graph_cfg.get("num_layers", 2),
        }

    agent = SACAgent(
        sample_obs=obs,
        action_shape=action_shape,
        action_low=action_low,
        action_high=action_high,
        cfg=agent_cfg,
        device=device,
    )
    replay = _build_replay(obs, action_shape, num_envs, config, device)

    # --- logging ------------------------------------------------------------
    logger = Logger(config["logdir"], config["logger"]["outputs"])
    if "wandb" in config["logger"]["outputs"]:
        import wandb
        wandb.init(
            project=config["logger"].get("wandb_project") or "sac",
            entity=config["logger"].get("wandb_entity") or None,
            name=config["logger"].get("wandb_name") or None,
            group=config["logger"].get("wandb_group") or None,
            config=config,
            dir=config["logdir"],
            resume="allow",
        )

    # --- training loop ------------------------------------------------------
    total_steps = int(run_cfg["total_steps"])
    learning_starts = int(agent_cfg["learning_starts"])
    training_freq = int(agent_cfg["training_freq"])
    utd = float(agent_cfg["utd"])
    grad_steps_per_block = max(1, int(training_freq * utd))
    steps_per_env_block = max(1, training_freq // num_envs)
    log_every = int(run_cfg["log_every"])
    eval_every = int(run_cfg["eval_every"])
    save_every = int(run_cfg["save_every"])
    eval_steps = int(run_cfg["eval_steps"])

    global_step = 0
    last_log = -log_every
    last_console = -log_every
    last_eval = -eval_every
    last_save = -save_every
    learning_has_started = False
    cumulative = defaultdict(float)

    print(f"[sac] task={task} obs_mode={obs_mode} num_envs={num_envs} "
          f"buffer={agent_cfg['buffer_size']} device={device}")
    print("[sac] Start training loop", flush=True)

    while global_step < total_steps:
        # ----- collect a training_freq block --------------------------------
        rollout_t = time.perf_counter()
        for _ in range(steps_per_env_block):
            if not learning_has_started:
                act = (
                    2.0 * torch.rand(envs.action_space.shape,
                                     dtype=torch.float32, device=device) - 1.0
                )
            else:
                act = agent.act(obs, deterministic=False)
            act = act.detach()

            next_obs_raw, reward, terminated, truncated, info = envs.step(act)
            next_obs = adapt_obs(next_obs_raw, obs_mode, device)
            done = (terminated | truncated)
            if graph_train is not None:
                next_obs.update(graph_train.step(done, device))

            # Real next-obs uses final_observation on done envs so the Q-target
            # bootstraps off the actual final state, not the auto-reset state.
            # Graph keys are exempt: the builder is stateful, so we cannot
            # rewind it to the pre-reset frame.
            real_next_obs = {k: v.clone() for k, v in next_obs.items()}
            bootstrap = str(agent_cfg["bootstrap_at_done"])
            if bootstrap == "always":
                need_final = done
                stop_bootstrap = torch.zeros_like(terminated, dtype=torch.bool)
            elif bootstrap == "truncated":
                need_final = truncated & (~terminated)
                stop_bootstrap = terminated
            elif bootstrap == "never":
                need_final = done
                stop_bootstrap = done
            else:
                raise ValueError(f"Unknown bootstrap_at_done={bootstrap!r}")

            if "final_observation" in info and need_final.any():
                final_raw = info["final_observation"]
                final_adapt = adapt_obs(final_raw, obs_mode, device)
                for k in real_next_obs:
                    if k in final_adapt:
                        real_next_obs[k][need_final] = final_adapt[k][need_final]

            replay.add(
                obs=obs,
                next_obs=real_next_obs,
                action=act,
                reward=reward.to(device).float(),
                done=stop_bootstrap.to(device).float(),
            )

            obs = next_obs
            global_step += num_envs

            # Track per-block episode metrics from the env wrapper.
            if "final_info" in info and "episode" in info["final_info"]:
                mask = info.get("_final_info", done)
                ep = info["final_info"]["episode"]
                for key, val in ep.items():
                    arr = val.detach().float().view(-1)
                    arr = arr[mask.view(-1)] if mask is not None else arr
                    if arr.numel() > 0:
                        cumulative[f"train/{key}"] += float(arr.mean().item())
                        cumulative[f"train/{key}_n"] += 1.0
        rollout_t = time.perf_counter() - rollout_t
        steps_in_block = num_envs * steps_per_env_block

        if global_step - last_console >= log_every:
            last_console = global_step
            console_metrics = {
                "step": global_step,
                "total": total_steps,
                "replay": len(replay),
            }
            if rollout_t > 0:
                console_metrics["fps/rollout"] = steps_in_block / rollout_t
            for key, total in list(cumulative.items()):
                if key.endswith("_n"):
                    continue
                n = cumulative.get(f"{key}_n", 0.0)
                if n > 0:
                    console_metrics[key] = total / n
            phase = "collect" if global_step < learning_starts else "train"
            print(f"[sac] {phase} | {_format_console_metrics(console_metrics)}",
                  flush=True)

        # ----- updates -------------------------------------------------------
        if global_step < learning_starts:
            continue
        learning_has_started = True
        update_t = time.perf_counter()
        last_metrics = None
        for _ in range(grad_steps_per_block):
            batch = replay.sample(int(agent_cfg["batch_size"]))
            last_metrics = agent.update(batch)
        update_t = time.perf_counter() - update_t

        # ----- logging -------------------------------------------------------
        if global_step - last_log >= log_every and last_metrics is not None:
            last_log = global_step
            logger.scalar("losses/qf_loss", last_metrics.qf_loss, global_step)
            logger.scalar("losses/qf1_value", last_metrics.qf1_value, global_step)
            logger.scalar("losses/qf2_value", last_metrics.qf2_value, global_step)
            logger.scalar("losses/actor_loss", last_metrics.actor_loss, global_step)
            logger.scalar("losses/alpha", last_metrics.alpha, global_step)
            logger.scalar("losses/alpha_loss", last_metrics.alpha_loss, global_step)
            logger.scalar("time/rollout_s", rollout_t, global_step)
            logger.scalar("time/update_s", update_t, global_step)
            if rollout_t > 0:
                logger.scalar("time/rollout_fps",
                              steps_in_block / rollout_t, global_step)
            # Emit averaged episode metrics, then reset cumulative bins.
            for key, total in list(cumulative.items()):
                if key.endswith("_n"):
                    continue
                n = cumulative.get(f"{key}_n", 0.0)
                if n > 0:
                    logger.scalar(key, total / n, global_step)
            cumulative.clear()
            print("[sac] update | " + _format_console_metrics({
                "step": global_step,
                "qf_loss": last_metrics.qf_loss,
                "actor_loss": last_metrics.actor_loss,
                "alpha": last_metrics.alpha,
                "update_s": update_t,
            }), flush=True)

        # ----- eval ----------------------------------------------------------
        if global_step - last_eval >= eval_every:
            last_eval = global_step
            print(f"[sac] eval start | step: {global_step}, "
                  f"eval_steps: {eval_steps}, num_eval_envs: {num_eval_envs}",
                  flush=True)
            metrics = _evaluate(agent, eval_envs, eval_obs, obs_mode, device,
                                 num_steps=eval_steps, graph_obs=graph_eval)
            for key, value in metrics.items():
                logger.scalar(f"eval/{key}", value, global_step)
            shown = _format_console_metrics({
                **metrics,
                "step": global_step,
            })
            print(f"Eval metrics | {shown or f'step: {global_step}'}",
                  flush=True)

        # ----- save ----------------------------------------------------------
        if global_step - last_save >= save_every:
            last_save = global_step
            path = os.path.join(config["logdir"], f"ckpt_{global_step}.pt")
            torch.save({"agent": agent.state_dict(),
                         "global_step": global_step,
                         "config": config}, path)
            print(f"[sac] saved {path}")

    final_path = os.path.join(config["logdir"], "ckpt_final.pt")
    torch.save({"agent": agent.state_dict(),
                 "global_step": global_step,
                 "config": config}, final_path)
    print(f"[sac] saved {final_path}")
    logger.close()
    envs.close()
    eval_envs.close()


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _evaluate(agent, eval_envs, eval_obs, obs_mode, device, *, num_steps: int,
               graph_obs=None):
    metrics = defaultdict(list)
    obs = eval_obs
    obs_raw, _ = eval_envs.reset()
    obs = adapt_obs(obs_raw, obs_mode, device)
    if graph_obs is not None:
        obs.update(graph_obs.reset(device))
    for _ in range(num_steps):
        act = agent.act(obs, deterministic=True)
        obs_raw, _, terminated, truncated, info = eval_envs.step(act)
        obs = adapt_obs(obs_raw, obs_mode, device)
        if graph_obs is not None:
            obs.update(graph_obs.step(terminated | truncated, device))
        if "final_info" in info and "episode" in info["final_info"]:
            ep = info["final_info"]["episode"]
            mask = info.get("_final_info", terminated | truncated)
            for key, val in ep.items():
                arr = val.detach().float().view(-1)[mask.view(-1)]
                if arr.numel() > 0:
                    metrics[key].append(arr.mean().item())
    out = {k: float(np.mean(v)) for k, v in metrics.items() if v}
    return out
