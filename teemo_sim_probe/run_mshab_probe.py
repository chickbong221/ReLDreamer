"""Run the TEEMO sim-probe on MS-HAB checkpoints.

Design:
  * Env built in obs_mode="rgb+depth+segmentation": the depth wrapper +
    framestack feed the policy the depth/state obs it was trained on, while
    RGB and segmentation ride along in unwrapped sensor_data for the probe.
  * Wrapper stack copied from make_env: FetchDepthObservationWrapper(cat_state)
    -> FrameStack(...) -> FetchActionWrapper -> ManiSkillVectorEnv
    -> VectorRecordEpisodeStatistics.
  * Policy: PPO and SAC checkpoints are auto-detected from config.yml.
  * Probe reads segmentation from env.unwrapped.get_obs()["sensor_data"]
    ["fetch_head"]["segmentation"] each step.

Usage:
    python -m teemo_sim_probe.run_mshab_probe \
        --ckpt-dir mshab_checkpoints/rl/tidy_house/pick/all \
        --steps 60 --video
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .configs.loader import load_config
from .core.graph_builder import GraphBuilder
from .core.mask_extractor import (
    read_unwrapped_sensor, depth_to_gray_rgb, depth_to_color_rgb,
)
from .viz.overlay import render_overlay
from .viz.graph_draw import render_graph
from .viz.video_writer import write_video
from .viz.palette import ColorMap
from .viz.eval_view import render_eval_view, save_image, SuccessTracker
from .adapters.policy_loader import load_policy, detect_algo
from .adapters.privileged_state import get_privileged_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir",
                   default="mshab_checkpoints/rl/tidy_house/pick/all",
                   help="dir with the PPO .pt (policy.pt / latest.pt / best.pt)")
    p.add_argument("--task", default=None,
                   help="MS-HAB task name. Defaults to ckpt path/config, then tidy_house.")
    p.add_argument("--subtask", default=None,
                   help="MS-HAB subtask. Defaults to ckpt path/config, then pick.")
    p.add_argument("--split", default=None,
                   help="MS-HAB split. Defaults to ckpt config, then train.")
    p.add_argument("--obj", default=None,
                   help="Task-plan object file stem, e.g. 024_bowl. "
                        "Defaults to ckpt path/config, then all.")
    p.add_argument("--plan-index", type=int, default=0,
                   help="which plan from the selected task-plan JSON to run")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--probe-cameras", nargs="+",
                   default=["fetch_head", "fetch_hand"],
                   help="cameras to build overlays/graphs for")
    p.add_argument("--save-every", type=int, default=20,
                   help="save outputs every N steps (policy still steps every frame)")
    p.add_argument("--width", type=int, default=256,
                   help="probe camera width (independent of policy obs)")
    p.add_argument("--height", type=int, default=256,
                   help="probe camera height (independent of policy obs)")
    p.add_argument("--overlay-size", type=float, default=6.0,
                   help="overlay figure size in inches (display size)")
    p.add_argument("--eval-view", action="store_true", default=True,
                   help="save third-person evaluation-camera frames")
    p.add_argument("--no-eval-view", dest="eval_view", action="store_false")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--fetch-action-wrapper", choices=["auto", "on", "off"],
                   default="auto",
                   help="apply MS-HAB FetchActionWrapper; auto matches MS-HAB default")
    p.add_argument("--free-head", action="store_true",
                   help="when FetchActionWrapper is enabled, do not zero head actions")
    p.add_argument("--include-goals", action="store_true")
    p.add_argument("--include-background", action="store_true",
                   help="keep scene_background actor (filtered by default)")
    p.add_argument("--include-static-scene", action="store_true",
                   help="keep static furniture / apartment props (filtered by default)")
    # Ablation flags (Track A: hard whitelist gate, no soft score).
    p.add_argument("--k-persist", type=int, default=None,
                   help="override selection.k_persist (frames)")
    p.add_argument("--n-slots", type=int, default=None,
                   help="override selection.n_slots")
    p.add_argument("--no-local-contact", action="store_true",
                   help="disable the local-contact one-hop expansion")
    p.add_argument("--whitelist-dir", default=None,
                   help="override the per-subtask whitelist directory "
                        "(defaults to configs/subtask_whitelists)")
    p.add_argument(
        "--mshab-object-name",
        choices=["actual", "merged"],
        default="actual",
        help="name active targets by their actual per-env actor name (default) "
             "or MS-HAB's merged obj_x name",
    )
    p.add_argument("--backdrop", choices=["rgb", "depth-color", "depth-gray"],
                   default="rgb",
                   help="overlay background: rgb (if available), turbo-colored "
                        "depth, or grayscale depth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "outputs", "mshab"))
    p.add_argument("--video", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    import gymnasium as gym
    import torch
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs  # noqa: F401  registers MS-HAB envs
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import FetchDepthObservationWrapper, FrameStack
    VectorRecordEpisodeStatistics = _load_vector_stats_wrapper()

    algo_hint = detect_algo(args.ckpt_dir)
    task, subtask, split, obj = _resolve_mshab_selection(args)
    print(f"[env] task={task} subtask={subtask} split={split} obj={obj}")
    print(f"[policy] config algo hint = {algo_hint}")

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / task / subtask / split / f"{obj}.json"
    if not plan_fp.exists():
        raise FileNotFoundError(
            f"MS-HAB task plan not found: {plan_fp}. "
            f"For object-specific checkpoints such as .../{obj}, the probe "
            f"must use the matching task-plan JSON. Pass --obj all only if "
            f"you intentionally want the aggregate plan file."
        )
    print(f"[env] task_plan={plan_fp}")
    plan_data = plan_data_from_file(plan_fp)
    if not (0 <= args.plan_index < len(plan_data.plans)):
        raise IndexError(
            f"--plan-index {args.plan_index} outside selected plan range "
            f"[0, {len(plan_data.plans) - 1}] for {plan_fp}"
        )
    spawn_data_fp = RD / "spawn_data" / task / subtask / split / "spawn_data.pt"

    env_id = f"{subtask.capitalize()}SubtaskTrain-v0"

    # Build the env in rgb+depth+segmentation. The depth wrapper only reads the
    # depth texture, so the policy obs is unchanged; the probe reads segmentation
    # (for masks) and rgb (for the overlay backdrop) from the unwrapped env.
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=200,
        task_plans=[plan_data.plans[args.plan_index]],
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=args.width, height=args.height),
    )

    # Wrapper stack copied from mshab.envs.make.make_env. Released Fetch
    # checkpoints use stationary_head=True, so keep the same action masking
    # unless the user explicitly disables it.
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    if args.frame_stack:
        env = FrameStack(
            env,
            num_stack=args.frame_stack,
            stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
        )
    use_fetch_action_wrapper = (
        args.fetch_action_wrapper == "on"
        or args.fetch_action_wrapper == "auto"
    )
    if use_fetch_action_wrapper:
        from mshab.envs.wrappers import FetchActionWrapper
        env = FetchActionWrapper(
            env,
            stationary_base=False,
            stationary_torso=False,
            stationary_head=not args.free_head,
        )
        print(
            "[env] FetchActionWrapper enabled "
            f"(stationary_head={not args.free_head})"
        )
    else:
        print("[env] FetchActionWrapper disabled")
    venv = ManiSkillVectorEnv(
        env,
        ignore_terminations=True,
        max_episode_steps=200,
    )
    venv = VectorRecordEpisodeStatistics(venv, max_episode_steps=200)

    eval_obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(args.ckpt_dir, venv, eval_obs, device=args.device)
    print(f"[policy] kind={policy.kind}")

    cfg = load_config("room_scale")
    _apply_ablation_overrides(cfg, args)
    cameras = list(args.probe_cameras)
    # One GraphBuilder per camera (separate temporal buffers + visibility).
    builders = {
        cam: GraphBuilder(
            venv, cfg, env_idx=0, env_id=env_id,
            camera=cam, include_goals=args.include_goals,
            include_background=args.include_background,
            include_static_scene=args.include_static_scene,
            mshab_object_name=args.mshab_object_name,
        )
        for cam in cameras
    }
    # One shared colormap per camera so object colors are stable within a camera.
    colormaps = {cam: ColorMap() for cam in cameras}

    success = SuccessTracker(env_idx=0)
    # collect saved frame paths per camera for optional video
    overlay_paths = {cam: [] for cam in cameras}
    graph_paths = {cam: [] for cam in cameras}

    obs = eval_obs
    info = {}
    just_reset = True       # first iter is the start of an episode
    for frame in range(args.steps):
        save_now = (frame % args.save_every == 0) or (frame == args.steps - 1)

        if save_now:
            for cam in cameras:
                seg, depth, rgb_sensor = read_unwrapped_sensor(
                    venv, cam, env_idx=0
                )
                if args.backdrop == "rgb" and rgb_sensor is not None:
                    backdrop = rgb_sensor
                elif args.backdrop == "depth-gray" and depth is not None:
                    backdrop = depth_to_gray_rgb(depth)
                elif depth is not None:
                    backdrop = depth_to_color_rgb(depth)
                else:
                    backdrop = None

                graph, masks, cam_name, rgb = builders[cam].step(
                    obs, frame,
                    episode_boundary=just_reset,
                    seg_override=seg, rgb_override=backdrop,
                    camera_override=cam,
                )

                graph.save(os.path.join(args.out, f"graph_{cam}_{frame:04d}.json"))
                op = render_overlay(
                    rgb, graph, masks,
                    os.path.join(args.out, f"overlay_{cam}_{frame:04d}.png"),
                    colormap=colormaps[cam], target_inches=args.overlay_size,
                )
                gp = render_graph(
                    graph, os.path.join(args.out, f"graph_{cam}_{frame:04d}.png"),
                    colormap=colormaps[cam],
                )
                overlay_paths[cam].append(op)
                graph_paths[cam].append(gp)

            # Third-person evaluation-camera frame.
            if args.eval_view:
                ev = render_eval_view(venv, env_idx=0)
                if ev is not None:
                    save_image(ev, os.path.join(
                        args.out, f"eval_view_{frame:04d}.png"))

            if frame == 0:
                _print_mshab_summary(
                    venv, cameras[0], args.mshab_object_name
                )

        action = policy.act(obs)
        obs, reward, terminated, truncated, info = venv.step(action)
        success.update(info, frame)
        # ManiSkillVectorEnv auto-resets internally on done; mark the next frame
        # as a fresh episode for the builders.
        import torch as _torch
        done_t = _torch.logical_or(
            _as_torch(terminated, device=_torch.device("cpu"), dtype=_torch.bool),
            _as_torch(truncated, device=_torch.device("cpu"), dtype=_torch.bool),
        )
        just_reset = bool(done_t[0].item()) if done_t.numel() > 0 else False

    # success_once line figure + csv.
    success.save_plot(os.path.join(args.out, "success_once.png"),
                      title=f"{env_id} success_once")
    success.save_csv(os.path.join(args.out, "success_once.csv"))
    print(f"[success] ever_succeeded={success.success_once[-1] if success.success_once else 0}")

    if args.video:
        for cam in cameras:
            if overlay_paths[cam]:
                vid = write_video(
                    overlay_paths[cam], graph_paths[cam],
                    os.path.join(args.out, f"probe_{cam}.mp4"), fps=2,
                )
                print(f"video[{cam}]:", vid)

    venv.close()
    print(f"wrote {args.steps} frames to {args.out}")


def _apply_ablation_overrides(cfg: dict, args) -> None:
    sel = cfg["selection"]
    if args.k_persist is not None:
        sel["k_persist"] = int(args.k_persist)
    if args.n_slots is not None:
        sel["n_slots"] = int(args.n_slots)
    if args.no_local_contact:
        sel["enable_local_contact"] = False
    if args.whitelist_dir is not None:
        cfg["whitelist_dir"] = args.whitelist_dir


def _print_mshab_summary(venv, camera, mshab_object_name="actual"):
    e = venv.unwrapped
    print(f"--- probe camera: {camera} ---")
    print("--- subtask handles ---")
    ptr = int(np.asarray(
        e.subtask_pointer.detach().cpu() if hasattr(e.subtask_pointer, "detach")
        else e.subtask_pointer)[0])
    print(f"  subtask_pointer[0] = {ptr}")
    objs = getattr(e, "subtask_objs", [])
    arts = getattr(e, "subtask_articulations", [])
    if ptr < len(objs):
        o = objs[ptr]
        merged_name = getattr(o, "name", None) if o is not None else None
        state = get_privileged_state(
            venv, env_idx=0, mshab_object_name=mshab_object_name
        )
        graph_name = (
            getattr(state.active_obj, "name", None)
            if state.active_obj is not None else None
        )
        print(f"  active_obj_merged = {merged_name}")
        print(
            f"  active_obj_graph = {graph_name} "
            f"(mode={mshab_object_name})"
        )
    if ptr < len(arts):
        a = arts[ptr]
        print(f"  active_articulation = "
              f"{getattr(a, 'name', None) if a is not None else None}")


def _load_vector_stats_wrapper():
    try:
        from mshab.envs.wrappers.vector import VectorRecordEpisodeStatistics
        return VectorRecordEpisodeStatistics
    except ImportError as exc:
        print(
            "[env] MS-HAB VectorRecordEpisodeStatistics import failed "
            f"({exc}); using local compatibility wrapper"
        )
        return _CompatVectorRecordEpisodeStatistics


class _CompatVectorRecordEpisodeStatistics:
    """Small compatibility replacement for older MS-HAB + newer Gymnasium."""

    def __init__(self, env, max_episode_steps: int = -1, extra_stat_keys=None):
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.extra_stat_keys = extra_stat_keys or []
        self.reset_queues()

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self):
        return self.unwrapped.num_envs

    def reset_queues(self):
        self.return_queue = []
        self.length_queue = []
        self.success_once_queue = []
        self.success_at_end_queue = []
        self.extra_stats = {k: [] for k in self.extra_stat_keys}

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        import torch
        device = getattr(self.unwrapped, "device", None)
        self.episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self.episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.int32, device=device
        )
        self.episode_success_onces = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.episode_success_at_ends = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        return obs, info

    def step(self, actions):
        import torch

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        device = self.episode_returns.device
        rewards_t = _as_torch(rewards, device=device, dtype=torch.float32)
        term_t = _as_torch(terminations, device=device, dtype=torch.bool)
        trunc_t = _as_torch(truncations, device=device, dtype=torch.bool)
        dones = torch.logical_or(term_t, trunc_t)

        self.episode_returns += rewards_t
        self.episode_lengths += 1

        success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        if "success" in infos:
            success = _as_torch(infos["success"], device=device, dtype=torch.bool)
            if "_success" in infos:
                success = torch.logical_and(
                    success,
                    _as_torch(infos["_success"], device=device, dtype=torch.bool),
                )
        if "_final_info" in infos and isinstance(infos.get("final_info"), dict):
            final_info = infos["final_info"]
            if "success" in final_info:
                final_success = _as_torch(
                    final_info["success"], device=device, dtype=torch.bool
                )
                final_mask = _as_torch(
                    infos["_final_info"], device=device, dtype=torch.bool
                )
                success = torch.where(final_mask, final_success, success)

        self.episode_success_at_ends = success
        self.episode_success_onces = torch.logical_or(
            self.episode_success_onces, self.episode_success_at_ends
        )

        if torch.any(dones):
            infos.pop("episode", None)
            infos.pop("_episode", None)
            infos["episode"] = {
                "r": torch.where(dones, self.episode_returns, 0.0),
                "l": torch.where(dones, self.episode_lengths, 0),
                "s_o": torch.where(dones, self.episode_success_onces, False),
                "s_e": torch.where(dones, self.episode_success_at_ends, False),
            }
            infos["_episode"] = dones

            for i in torch.where(dones)[0].tolist():
                self.return_queue.append(self.episode_returns[i])
                self.length_queue.append(self.episode_lengths[i])
                self.success_once_queue.append(self.episode_success_onces[i])
                self.success_at_end_queue.append(self.episode_success_at_ends[i])

            self.episode_returns[dones] = 0.0
            self.episode_lengths[dones] = 0
            self.episode_success_onces[dones] = False
            self.episode_success_at_ends[dones] = False

        return observations, rewards, terminations, truncations, infos

    def close(self):
        return self.env.close()


def _as_torch(value, device, dtype):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _resolve_mshab_selection(args):
    cfg = _load_ckpt_config(args.ckpt_dir)
    task_p, subtask_p, obj_p = _infer_selection_from_ckpt_path(args.ckpt_dir)
    task = args.task or _find_config_value(
        cfg, ("mshab_task", "hab_task", "task_name")
    ) or task_p or "tidy_house"
    subtask = args.subtask or _find_config_value(
        cfg, ("subtask", "subtask_name")
    ) or subtask_p or "pick"
    split = args.split or _find_config_value(
        cfg, ("mshab_split", "split", "eval_split")
    ) or "train"
    obj = args.obj or _find_config_value(
        cfg, ("mshab_obj", "obj", "object", "object_id", "task_plan")
    ) or obj_p or "all"

    # Some configs store the Gym env id as "task"; prefer the explicit subtask
    # parsed from it when no --subtask was passed.
    if args.subtask is None:
        env_id = _find_config_value(cfg, ("env_id", "task"))
        inferred = _subtask_from_env_id(env_id)
        if inferred:
            subtask = inferred

    return str(task), str(subtask), str(split), _clean_obj_name(str(obj))


def _load_ckpt_config(ckpt_dir):
    cfg_path = ckpt_dir
    if os.path.isfile(cfg_path):
        cfg_path = os.path.dirname(cfg_path)
    cfg_path = os.path.join(cfg_path, "config.yml")
    if not os.path.exists(cfg_path):
        return {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _infer_selection_from_ckpt_path(ckpt_dir):
    path = ckpt_dir
    if os.path.isfile(path):
        path = os.path.dirname(path)
    parts = os.path.normpath(path).split(os.sep)
    if "rl" in parts:
        i = parts.index("rl")
        if len(parts) > i + 3:
            task = parts[i + 1]
            subtask = parts[i + 2]
            obj = parts[i + 3]
            return task, subtask, _clean_obj_name(obj)
    if len(parts) >= 3:
        return None, parts[-2], _clean_obj_name(parts[-1])
    return None, None, None


def _find_config_value(value, keys):
    if isinstance(value, dict):
        for key in keys:
            if key in value and value[key] not in (None, ""):
                return value[key]
        for child in value.values():
            found = _find_config_value(child, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_config_value(child, keys)
            if found not in (None, ""):
                return found
    return None


def _subtask_from_env_id(env_id):
    if not env_id:
        return None
    text = str(env_id)
    suffix = "SubtaskTrain-v0"
    if text.endswith(suffix):
        return text[: -len(suffix)].lower()
    return None


def _clean_obj_name(obj):
    if obj.endswith(".json"):
        return obj[:-5]
    return obj


if __name__ == "__main__":
    main()
