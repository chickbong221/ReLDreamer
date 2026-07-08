"""SAC training loop ported from mshab/train_sac.py.

One iteration = one env-step block of ``num_envs`` steps + one gradient
update. On the first crossing of ``init_steps`` the update runs
``init_steps`` times in one shot (mshab burn-in).

Eval runs a full ``eval_max_episode_steps``-length sweep across all eval envs.
When graph is enabled and wandb is on, one ``head_overlay | hand_overlay |
graph`` mp4 is rendered per recorded env (envs 0 and 1 by default) and pushed
to wandb. Hand panel is omitted when only one camera is configured.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional

from gymnasium import spaces

import numpy as np
import torch

from ..agent import SACAgent, UpdateMetrics
from ..envs import action_box, adapt_obs, build_env
from ..graph_env import build_graph_obs
from ..replay import PixelStateBatchReplayBuffer


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Logger:
    def __init__(self, logdir: str, outputs):
        self.logdir = logdir
        self.outputs = list(outputs)
        os.makedirs(logdir, exist_ok=True)
        self.tb = None
        self.jsonl_path = None
        self.wandb = None
        if "tensorboard" in self.outputs:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(logdir)
        if "jsonl" in self.outputs:
            self.jsonl_path = os.path.join(logdir, "metrics.jsonl")
        if "wandb" in self.outputs:
            import wandb
            self.wandb = wandb

    def scalar(self, tag: str, value, step: int) -> None:
        v = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        if self.tb is not None:
            self.tb.add_scalar(tag, v, step)
        if self.wandb is not None:
            self.wandb.log({tag: v}, step=step)
        if self.jsonl_path is not None:
            import json
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps({"step": step, "tag": tag, "value": v}) + "\n")

    def video(self, tag: str, path: str, step: int) -> None:
        if self.wandb is not None:
            self.wandb.log(
                {tag: self.wandb.Video(str(path), format="mp4")}, step=step,
            )

    def close(self):
        if self.tb is not None:
            self.tb.close()


_LIBC = None


def _trim_native_heap() -> bool:
    """Return freed-but-retained glibc heap pages to the OS (Linux only).

    Small-object churn grows arenas glibc rarely shrinks, so RSS climbs
    linearly with zero Python-level retention; malloc_trim(0) releases it.
    Returns whether the trim actually ran, so logs show if it is inert
    (e.g. non-Linux platform).
    """
    global _LIBC
    if not sys.platform.startswith("linux"):
        return False
    if _LIBC is None:
        import ctypes
        try:
            _LIBC = ctypes.CDLL("libc.so.6")
        except OSError:
            return False
    try:
        _LIBC.malloc_trim(0)
        return True
    except Exception:
        return False


class _RamLogger:
    """Append process + system RAM to ``ram_usage.csv`` so the leak slope is
    inspectable offline; also mirrors to the metrics logger."""

    def __init__(self, logdir: str):
        self.path = os.path.join(logdir, "ram_usage.csv")
        self._t0 = time.time()
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None
        with open(self.path, "w") as f:
            f.write("step,elapsed_s,proc_rss_mb,sys_used_mb,sys_avail_mb\n")

    def log(self, logger: "Logger", step: int):
        if self._proc is None:
            return None
        import psutil
        rss = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        vm = psutil.virtual_memory()
        rss_mb, used_mb, avail_mb = rss / 2**20, vm.used / 2**20, vm.available / 2**20
        with open(self.path, "a") as f:
            f.write(f"{step},{time.time() - self._t0:.1f},{rss_mb:.1f},"
                    f"{used_mb:.1f},{avail_mb:.1f}\n")
        logger.scalar("sys/proc_rss_mb", rss_mb, step)
        logger.scalar("sys/sys_used_mb", used_mb, step)
        logger.scalar("sys/sys_avail_mb", avail_mb, step)
        return rss_mb, avail_mb


def train(config: dict) -> None:
    suite, task = config["task"].split("_", 1)
    if suite != "maniskill":
        raise NotImplementedError(
            f"sac.run.train only supports 'maniskill' (got {suite!r})."
        )

    env_cfg = dict(config["env"]["maniskill"])
    agent_cfg = dict(config["agent"])
    run_cfg = dict(config["run"])
    graph_raw = dict(config.get("graph", {}))
    graph_enabled = bool(graph_raw.get("enabled", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(config["seed"]))

    train_split = env_cfg["mshab_split"]
    eval_split = env_cfg.get("mshab_eval_split") or train_split
    print(f"[sac] mshab split: train='{train_split}' eval='{eval_split}' "
          f"task={env_cfg['mshab_task']} obj={env_cfg['mshab_obj']} "
          f"nonprivileged_obs={bool(env_cfg.get('nonprivileged_obs', True))}",
          flush=True)
    envs = build_env(task, env_cfg, is_eval=False, seed=int(config["seed"]),
                     graph_enabled=graph_enabled)
    eval_envs = build_env(task, env_cfg, is_eval=True, seed=int(config["seed"]) + 1,
                          graph_enabled=graph_enabled)
    action_shape, _, _ = action_box(envs)
    num_envs = int(env_cfg["num_envs"])
    num_eval_envs = int(env_cfg["num_eval_envs"])

    graph_train = build_graph_obs(envs, graph_raw, num_envs=num_envs)
    graph_eval = build_graph_obs(eval_envs, graph_raw, num_envs=num_eval_envs)

    raw_obs, _ = envs.reset(seed=int(config["seed"]))
    obs = adapt_obs(raw_obs, device)
    graph_obs = graph_train.reset(device) if graph_train is not None else None
    eval_envs.reset(seed=int(config["seed"]) + 1)

    replay_pixels_space, agent_pixels_space = _pixels_obs_spaces(obs["pixels"])
    state_dim = int(obs["state"].shape[1])

    graph_agent_cfg: Optional[dict] = None
    if graph_enabled:
        graph_agent_cfg = dict(
            node_vocab_size=len(graph_train.node_vocab),
            edge_vocab_size=len(graph_train.edge_vocab),
            embed_dim=int(graph_raw.get("embed_dim", 64)),
            hidden_dim=int(graph_raw.get("hidden_dim", 512)),
            out_dim=int(graph_raw.get("out_dim", 128)),
            num_layers=int(graph_raw.get("num_layers", 2)),
            actor_gradient=bool(graph_raw.get("actor_gradient", False)),
        )

    agent = SACAgent(
        pixels_obs_space=agent_pixels_space,
        state_dim=state_dim,
        action_shape=action_shape,
        cfg=agent_cfg,
        graph_cfg=graph_agent_cfg,
        device=device,
    )

    horizon = int(env_cfg["max_episode_steps"])
    buffer_cfg = int(agent_cfg["buffer_size"])
    capacity = (buffer_cfg // (num_envs * horizon)) * (num_envs * horizon)
    assert capacity > 0, (
        f"buffer_size={buffer_cfg} too small for num_envs*horizon={num_envs*horizon}"
    )
    replay_horizon = horizon - 1
    replay_size = (capacity // horizon) * replay_horizon

    graph_shapes = graph_dtypes = None
    if graph_enabled:
        graph_shapes = graph_train.obs_spec_shapes
        graph_dtypes = {k: v.detach().cpu().numpy().dtype for k, v in graph_obs.items()}
    replay = PixelStateBatchReplayBuffer(
        pixels_obs_space=replay_pixels_space,
        state_obs_dim=state_dim,
        act_dim=int(np.prod(action_shape)),
        size=replay_size,
        horizon=replay_horizon,
        num_envs=num_envs,
        graph_shapes=graph_shapes,
        graph_dtypes=graph_dtypes,
    )

    logger = Logger(config["logdir"], config["logger"]["outputs"])
    ram_logger = _RamLogger(config["logdir"])
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

    total_steps = int(run_cfg["total_steps"])
    init_steps = int(agent_cfg["init_steps"])
    log_every = int(run_cfg["log_every"])
    eval_every = int(run_cfg["eval_every"])
    save_every = int(run_cfg["save_every"])
    eval_max_steps = int(env_cfg["eval_max_episode_steps"])
    batch_size = int(agent_cfg["batch_size"])

    global_step = 0
    last_log = -log_every
    last_eval = -eval_every
    last_save = 0
    num_iterations = math.ceil(total_steps / num_envs)

    print(f"[sac] task={task} num_envs={num_envs} horizon={horizon} "
          f"buffer(cap)={capacity} buffer(trans)={replay_size} device={device}",
          flush=True)

    for iteration in range(num_iterations):
        if global_step >= total_steps:
            break

        if len(replay) < init_steps:
            act = 2.0 * torch.rand(envs.action_space.shape,
                                    dtype=torch.float32, device=device) - 1.0
        else:
            act = agent.act(obs["pixels"], obs["state"], graph_obs, deterministic=False)

        next_raw, rew, term, trunc, _ = envs.step(act)
        next_obs = adapt_obs(next_raw, device)
        done = term | trunc
        graph_next = graph_train.step(done, device) if graph_train is not None else None

        trunc_any = bool(trunc.any().item())
        replay.store_batch(
            pixel_obs={k: v.detach().cpu().numpy().astype(np.uint16) for k, v in obs["pixels"].items()},
            pixel_next_obs={k: v.detach().cpu().numpy().astype(np.uint16) for k, v in next_obs["pixels"].items()},
            state_obs=obs["state"].detach().cpu().numpy(),
            state_next_obs=next_obs["state"].detach().cpu().numpy(),
            act=act.detach().cpu().numpy(),
            rew=rew.detach().cpu().numpy(),
            term=term.detach().cpu().numpy().astype(np.float32),
            graph_obs=(
                {k: v.detach().cpu().numpy() for k, v in graph_obs.items()}
                if graph_obs is not None else None
            ),
            graph_next_obs=(
                {k: v.detach().cpu().numpy() for k, v in graph_next.items()}
                if graph_next is not None else None
            ),
            trunc_any=trunc_any,
        )

        obs, graph_obs = next_obs, graph_next
        global_step += num_envs

        loss_metrics: Optional[UpdateMetrics] = None
        if len(replay) >= init_steps:
            num_updates = init_steps if len(replay) == init_steps else 1
            for _ in range(num_updates):
                batch = _batch_to_tensors(replay.sample_batch(batch_size), device)
                loss_metrics = agent.update(batch)

        if global_step - last_log >= log_every:
            last_log = global_step
            if len(envs.return_queue) > 0:
                _log_env_stats(logger, envs, "train", global_step)
            if loss_metrics is not None:
                _log_losses(logger, loss_metrics, global_step)
            trimmed = _trim_native_heap()
            ram = ram_logger.log(logger, global_step)
            ram_str = (f" rss={ram[0] / 1024:.1f}GB avail={ram[1] / 1024:.1f}GB"
                       if ram else "")
            print(f"[sac] step={global_step}/{total_steps} "
                  f"replay={len(replay)}{ram_str}", flush=True)
            if graph_train is not None:
                import gc
                stats = graph_train.cache_stats()
                stats["py_objects"] = len(gc.get_objects())
                stats["malloc_trim"] = int(trimmed)
                for k, v in stats.items():
                    logger.scalar(f"leak/{k}", v, global_step)
                print("[leak] " + " ".join(f"{k}={v}" for k, v in stats.items()),
                      flush=True)

        if eval_every > 0 and global_step - last_eval >= eval_every:
            last_eval = global_step
            _run_eval(
                agent, eval_envs, graph_eval, logger, device,
                eval_max_steps=eval_max_steps,
                step=global_step,
                wandb_enabled="wandb" in config["logger"]["outputs"],
            )

        if save_every > 0 and global_step - last_save >= save_every:
            last_save = global_step
            torch.save(
                {"agent": agent.state_dict(),
                 "log_alpha": agent.log_alpha.detach()},
                os.path.join(config["logdir"], "latest.pt"),
            )

    logger.close()


def _pixels_obs_spaces(pixels_obs: Dict[str, torch.Tensor]):
    """Return (replay_space, agent_space).

    replay: 4D per-camera (fs, C, H, W) so PixelStateBatchReplayBuffer can
    materialize frame stacks at sample time.
    agent: 3D per-camera (fs*C, H, W) for SharedCNN's Conv2d input.
    """
    replay_dict, agent_dict = {}, {}
    for k, v in pixels_obs.items():
        _, fs, c, h, w = v.shape
        replay_dict[k] = spaces.Box(low=0, high=65535, shape=(fs, c, h, w), dtype=np.uint16)
        agent_dict[k] = spaces.Box(low=0, high=65535, shape=(fs * c, h, w), dtype=np.uint16)
    return spaces.Dict(replay_dict), spaces.Dict(agent_dict)


def _batch_to_tensors(batch, device):
    to_f = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    out = dict(
        pixel_obs={k: to_f(v) for k, v in batch["pixel_obs"].items()},
        pixel_next_obs={k: to_f(v) for k, v in batch["pixel_next_obs"].items()},
        state_obs=to_f(batch["state_obs"]),
        state_next_obs=to_f(batch["state_next_obs"]),
        act=to_f(batch["act"]),
        rew=to_f(batch["rew"]),
        term=to_f(batch["term"]),
    )
    if "graph_obs" in batch:
        out["graph_obs"] = {k: torch.as_tensor(v, device=device) for k, v in batch["graph_obs"].items()}
        out["graph_next_obs"] = {k: torch.as_tensor(v, device=device) for k, v in batch["graph_next_obs"].items()}
    return out


def _log_env_stats(logger: Logger, env, key: str, step: int) -> None:
    def _mean(queue) -> float:
        vals = [x.detach().float() if isinstance(x, torch.Tensor) else torch.as_tensor(x).float()
                 for x in queue]
        return float(torch.stack(vals).mean().item()) if vals else 0.0
    logger.scalar(f"{key}/return_per_step",
                   _mean(env.return_queue) / env.max_episode_steps, step)
    logger.scalar(f"{key}/success_once", _mean(env.success_once_queue), step)
    logger.scalar(f"{key}/success_at_end", _mean(env.success_at_end_queue), step)
    logger.scalar(f"{key}/len", _mean(env.length_queue), step)
    env.reset_queues()


def _log_losses(logger: Logger, m: UpdateMetrics, step: int) -> None:
    logger.scalar("losses/critic_loss", m.critic_loss, step)
    logger.scalar("losses/q1", m.q1, step)
    logger.scalar("losses/q2", m.q2, step)
    if m.actor_loss is not None:
        logger.scalar("losses/actor_loss", m.actor_loss, step)
        logger.scalar("losses/alpha_loss", m.alpha_loss, step)
        logger.scalar("losses/entropy_mean", m.entropy_mean, step)
    logger.scalar("alpha", m.alpha, step)
    logger.scalar("log_alpha", m.log_alpha, step)
    logger.scalar("target_entropy", m.target_entropy, step)


def _run_eval(agent, eval_envs, graph_eval, logger, device, *,
               eval_max_steps: int, step: int, wandb_enabled: bool) -> None:
    do_video = graph_eval is not None and wandb_enabled
    tmp_dir = None
    hand_cam: Optional[str] = None
    record_ids: List[int] = []
    head_paths: Dict[int, list] = {}
    hand_paths: Dict[int, list] = {}
    graph_paths: Dict[int, list] = {}
    env_done: Dict[int, bool] = {}

    try:
        if do_video:
            from teemo_sim_probe.viz.palette import ColorMap
            from teemo_sim_probe.viz.overlay import render_overlay
            from teemo_sim_probe.viz.graph_draw import render_graph
            from teemo_sim_probe.core.mask_extractor import MaskAccumulator
            record_ids = [i for i in (0, 1) if i < graph_eval.num_envs]
            graph_eval.record_env_indices = set(record_ids)
            head_paths = {i: [] for i in record_ids}
            hand_paths = {i: [] for i in record_ids}
            graph_paths = {i: [] for i in record_ids}
            env_done = {i: False for i in record_ids}
            tmp_dir = tempfile.TemporaryDirectory()
            cmap = ColorMap()
            hand_cam = graph_eval.secondary_camera

        eval_raw, _ = eval_envs.reset()
        obs = adapt_obs(eval_raw, device)
        graph_obs = graph_eval.reset(device) if graph_eval is not None else None

        def _masks_from_seg(graph, seg: np.ndarray) -> "MaskAccumulator":
            H, W = seg.shape
            masks = MaskAccumulator(H, W)
            for node in graph.nodes:
                if not node.valid_mask or not node.segmentation_ids:
                    continue
                m = np.isin(seg, node.segmentation_ids)
                if bool(m.any()):
                    masks.add(node.node_id, m)
            return masks

        def _record_frame(env_idx: int, t: int) -> None:
            g = graph_eval.last_graph_by_env.get(env_idx)
            m = graph_eval.last_masks_by_env.get(env_idx)
            if g is None or m is None:
                return
            rgb_head = graph_eval.read_rgb(env_idx)
            head_paths[env_idx].append(render_overlay(
                rgb_head, g, m,
                os.path.join(tmp_dir.name, f"head_env{env_idx}_{t:04d}.png"),
                colormap=cmap,
            ))
            if hand_cam is not None:
                rgb_hand, seg_hand = graph_eval.read_view(hand_cam, env_idx)
                hand_masks = _masks_from_seg(g, seg_hand)
                hand_paths[env_idx].append(render_overlay(
                    rgb_hand, g, hand_masks,
                    os.path.join(tmp_dir.name, f"hand_env{env_idx}_{t:04d}.png"),
                    colormap=cmap,
                ))
            graph_paths[env_idx].append(render_graph(
                g,
                os.path.join(tmp_dir.name, f"graph_env{env_idx}_{t:04d}.png"),
                colormap=cmap,
            ))

        if do_video:
            for i in record_ids:
                _record_frame(i, 0)

        for t in range(eval_max_steps):
            with torch.no_grad():
                act = agent.act(
                    obs["pixels"], obs["state"], graph_obs, deterministic=True,
                )
            next_raw, _, term, trunc, _ = eval_envs.step(act)
            obs = adapt_obs(next_raw, device)
            done = term | trunc
            if graph_eval is not None:
                graph_obs = graph_eval.step(done, device)

            if do_video:
                for i in record_ids:
                    if env_done[i]:
                        continue
                    if bool(done[i].item()):
                        env_done[i] = True
                    else:
                        _record_frame(i, t + 1)

        if len(eval_envs.return_queue) > 0:
            _log_env_stats(logger, eval_envs, "eval", step)

        if do_video:
            from teemo_sim_probe.viz.video_writer import write_video
            for i in record_ids:
                if not head_paths[i]:
                    continue
                mp4 = os.path.join(tmp_dir.name, f"eval_env{i}.mp4")
                panels = [head_paths[i]]
                if hand_cam is not None:
                    panels.append(hand_paths[i])
                panels.append(graph_paths[i])
                write_video(panels, mp4, fps=5)
                logger.video(f"eval/graph_video/env{i}", mp4, step)
    finally:
        if do_video:
            graph_eval.record_env_indices = set()
            graph_eval.last_graph_by_env.clear()
            graph_eval.last_masks_by_env.clear()
        if tmp_dir is not None:
            tmp_dir.cleanup()
