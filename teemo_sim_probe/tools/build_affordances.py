"""Offline miner: success rollouts -> ``affordances.json``.

Per sample: ``anchor_obj = inv(obj_pose_wrt_base) * tcp_in_base``,
``width = qpos[-2] + qpos[-1]``, ``approach_dir_obj = inv_rot(obj) * R_tcp @ axis_local``.
Reads schema-v4 ``tcp_pose_wrt_base`` directly; falls back to SAPIEN FK on Fetch
for older pickles. PLACE is excluded: its success requires the TCP at the rest
pose (mshab/envs/sequential_task.py:1138), so the anchor would learn the rest pose.

Usage::

    python -m teemo_sim_probe.tools.build_affordances \\
        --success-states-dir $MS_ASSET_DIR/data/robot_success_states \\
        --out teemo_sim_probe/configs/affordances.json --robot fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


log = logging.getLogger("build_affordances")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _canonical_key(name: Optional[str]) -> Optional[str]:
    """Delegates to ``canonical_affordance_key`` so the two cannot drift."""
    if not name:
        return None
    value = str(name)
    if value.startswith(("actor:", "link:", "object:")):
        return value
    from teemo_sim_probe.core.affordance import canonical_affordance_key
    canonical = canonical_affordance_key(value)
    return f"actor:{canonical}" if canonical else None


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    qn = _normalize(np.asarray(q, dtype=float))
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _inv_transform_point(
    pose_wxyz: np.ndarray, point: np.ndarray
) -> Optional[np.ndarray]:
    """``inv(pose) * point`` for pose ``[xyz, qw, qx, qy, qz]``."""
    if pose_wxyz is None or len(pose_wxyz) < 7:
        return None
    p = np.asarray(pose_wxyz[:3], dtype=float)
    q = np.asarray(pose_wxyz[3:7], dtype=float)
    R = _quat_wxyz_to_rotmat(q)
    if R is None or not np.all(np.isfinite(p)) or not np.all(np.isfinite(point)):
        return None
    return R.T @ (np.asarray(point, dtype=float) - p)


def _inv_rotate_dir(
    pose_wxyz: np.ndarray, dir_world: np.ndarray
) -> Optional[np.ndarray]:
    """World direction -> OBJECT frame unit (``R_obj.T @ d``)."""
    if pose_wxyz is None or len(pose_wxyz) < 7:
        return None
    R = _quat_wxyz_to_rotmat(np.asarray(pose_wxyz[3:7], dtype=float))
    if R is None:
        return None
    d = R.T @ np.asarray(dir_world, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9 or not np.all(np.isfinite(d)):
        return None
    return d / n


# SAPIEN-based FK, Fetch only.
class _FetchFK:
    """Loads the Fetch URDF once; ``set_qpos`` + ``gripper_link.pose`` per sample.
    Root is fixed at identity so world == base frame."""

    def __init__(self, urdf_path: str, tcp_link_name: str = "gripper_link"):
        # Local imports so ``--help`` works without SAPIEN installed.
        try:
            import sapien                                       # noqa: F401
            from sapien import physx                            # noqa: F401
            import sapien.render                                # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "SAPIEN is required for FK. Install with the project's normal "
                "ManiSkill setup. Underlying error: " + repr(exc)
            ) from exc

        import sapien
        from sapien import physx
        import sapien.render

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Fetch URDF not found at {urdf_path}")

        self._sys = physx.PhysxCpuSystem()
        # URDF loader requires a render system even in headless FK
        # (see ManiSkill/mani_skill/envs/sapien_env.py:1213-1217).
        try:
            render_sys = sapien.render.RenderSystem()
        except Exception as exc:
            raise RuntimeError(
                "sapien.render.RenderSystem() failed -- it's required by the "
                f"URDF loader even in headless FK. Original error: {exc!r}"
            ) from exc
        self._scene = sapien.Scene([self._sys, render_sys])
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load(urdf_path)
        if self._robot is None:
            raise RuntimeError(f"Failed to load URDF at {urdf_path}")
        try:
            self._robot.set_root_pose(sapien.Pose())
        except Exception:
            pass  # Some SAPIEN versions lack set_root_pose; fix_root_link suffices.

        self._tcp_link = None
        for link in self._robot.get_links():
            if getattr(link, "name", None) == tcp_link_name:
                self._tcp_link = link
                break
        if self._tcp_link is None:
            raise RuntimeError(
                f"{tcp_link_name!r} not found in URDF links: "
                + ", ".join(getattr(l, "name", "?") for l in self._robot.get_links())
            )

        # qpos length may differ from MS-HAB's if joints are fixed/mimic.
        try:
            self._qpos_dim = int(self._robot.dof)
        except Exception:
            self._qpos_dim = -1

    def tcp_in_base(self, qpos: np.ndarray) -> Optional[np.ndarray]:
        """TCP pose ``[xyz, qw, qx, qy, qz]`` in base (= world) frame."""
        try:
            q = np.asarray(qpos, dtype=float).reshape(-1)
            if self._qpos_dim > 0 and q.shape[0] != self._qpos_dim:
                if q.shape[0] < self._qpos_dim:
                    return None
                q = q[: self._qpos_dim]
            self._robot.set_qpos(q)
            # Some SAPIEN versions need an explicit kinematic update.
            for refresh in ("compute_forward_kinematics",
                            "compute_kinematic_pass"):
                fn = getattr(self._robot, refresh, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
            pose = self._tcp_link.pose
            p = np.asarray(pose.p, dtype=float).reshape(-1)[:3]
            qw = np.asarray(pose.q, dtype=float).reshape(-1)[:4]
            out = np.concatenate([p, qw])
            if not np.all(np.isfinite(out)):
                return None
            return out
        except Exception as exc:
            log.debug("FK failure: %r", exc)
            return None


def _default_fetch_urdf() -> Optional[str]:
    """``fetch.urdf`` via ManiSkill's PACKAGE_ASSET_DIR, or None."""
    try:
        from mani_skill import PACKAGE_ASSET_DIR
    except Exception:
        return None
    return os.path.join(PACKAGE_ASSET_DIR, "robots", "fetch", "fetch.urdf")


# --------------------------------------------------------------------------- #
# Per-object mining
# --------------------------------------------------------------------------- #
def _load_pkl(path: Path) -> Optional[Dict]:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        log.error("failed to read %s: %r", path, exc)
        return None
    if not isinstance(data, dict):
        log.error("unexpected pkl shape in %s: %s", path, type(data))
        return None
    required = ("obj_id", "entity_key", "robot_qpos", "obj_pose_wrt_base")
    if not all(r in data for r in required):
        log.error("missing keys in %s: have=%s, need=%s",
                  path, sorted(data), required)
        return None
    # Prefer schema-v4 simulator TCP poses. FK is only a legacy fallback for
    # older pickles that do not carry tcp_pose_wrt_base.
    return data


def _mine_object(
    pkl_path: Path, fk: Optional[_FetchFK], max_samples: int,
    approach_axis_local: np.ndarray,
) -> Optional[Dict]:
    data = _load_pkl(pkl_path)
    if data is None:
        return None

    qpos_list = np.asarray(data["robot_qpos"], dtype=float)
    pose_list = np.asarray(data["obj_pose_wrt_base"], dtype=float)
    tcp_in_base_list: Optional[np.ndarray] = None
    if "tcp_pose_wrt_base" in data and data["tcp_pose_wrt_base"]:
        tcp_in_base_list = np.asarray(data["tcp_pose_wrt_base"], dtype=float)
    raw_obj_id = data.get("entity_key")
    canonical = _canonical_key(raw_obj_id)
    if canonical is None:
        log.error("%s: entity_key %r did not canonicalize", pkl_path, raw_obj_id)
        return None

    if qpos_list.ndim != 2 or pose_list.ndim != 2:
        log.error("%s: expected 2D arrays, got qpos=%s pose=%s",
                  pkl_path, qpos_list.shape, pose_list.shape)
        return None
    if qpos_list.shape[0] != pose_list.shape[0]:
        log.error("%s: qpos and pose lengths disagree (%d vs %d)",
                  pkl_path, qpos_list.shape[0], pose_list.shape[0])
        return None
    if pose_list.shape[1] < 7:
        log.error("%s: obj_pose_wrt_base needs 7-D rows, got %d",
                  pkl_path, pose_list.shape[1])
        return None
    if qpos_list.shape[1] < 2:
        log.error("%s: qpos width %d < 2 (need gripper joints)",
                  pkl_path, qpos_list.shape[1])
        return None
    if tcp_in_base_list is not None:
        if (
            tcp_in_base_list.ndim != 2
            or tcp_in_base_list.shape[0] != qpos_list.shape[0]
            or tcp_in_base_list.shape[1] < 7
        ):
            log.error(
                "%s: tcp_pose_wrt_base shape %s incompatible with qpos %s",
                pkl_path, tcp_in_base_list.shape, qpos_list.shape,
            )
            return None
    elif fk is None:
        log.error(
            "%s: no tcp_pose_wrt_base (schema-v4) and no FK fallback available",
            pkl_path,
        )
        return None

    n_samples = qpos_list.shape[0]
    if n_samples == 0:
        log.warning("%s: 0 samples, skipping", pkl_path)
        return None

    # Subsample if too many.
    if 0 < max_samples < n_samples:
        sel = np.random.default_rng(0).choice(n_samples, size=max_samples, replace=False)
        qpos_list = qpos_list[sel]
        pose_list = pose_list[sel]
        if tcp_in_base_list is not None:
            tcp_in_base_list = tcp_in_base_list[sel]
        n_samples = max_samples

    anchors_obj: List[np.ndarray] = []
    widths: List[float] = []
    approaches_obj: List[Optional[np.ndarray]] = []
    fk_failures = 0
    for i in range(n_samples):
        qpos = qpos_list[i]
        obj_pose = pose_list[i, :7]
        if tcp_in_base_list is not None:
            tcp_in_base = tcp_in_base_list[i, :7]
        else:
            tcp_in_base = fk.tcp_in_base(qpos)
            if tcp_in_base is None:
                fk_failures += 1
                continue
        anchor = _inv_transform_point(obj_pose, tcp_in_base[:3])
        if anchor is None or not np.all(np.isfinite(anchor)):
            continue
        # World approach direction = R_tcp @ axis_local; express in object frame.
        R_tcp = _quat_wxyz_to_rotmat(tcp_in_base[3:7])
        approach_obj: Optional[np.ndarray] = None
        if R_tcp is not None:
            dir_world = R_tcp @ approach_axis_local
            approach_obj = _inv_rotate_dir(obj_pose, dir_world)
        width = float(qpos[-2] + qpos[-1])
        if not np.isfinite(width):
            continue
        anchors_obj.append(anchor)
        widths.append(width)
        approaches_obj.append(approach_obj)   # may be None for this sample

    if fk_failures:
        log.warning("%s: FK failed on %d/%d samples", pkl_path, fk_failures, n_samples)

    if not anchors_obj:
        log.error("%s: no usable success samples -- skipping", pkl_path)
        return None

    components: List[Dict] = []
    for idx, (anchor, width, approach_obj) in enumerate(
        zip(anchors_obj, widths, approaches_obj)
    ):
        approach_field: Dict[str, List[float]] = {}
        if approach_obj is not None:
            n = float(np.linalg.norm(approach_obj))
            if n > 1e-9:
                approach_field = {
                    "approach_dir": [round(float(x), 6) for x in (approach_obj / n)]
                }
        components.append({
            "anchor": [round(float(x), 6) for x in anchor],
            **approach_field,
            "width": round(float(width), 6),
            "sample_index": int(idx),
        })

    log.info("%s -> %s : %d raw affordance candidates",
             pkl_path.name, canonical, len(components))
    return {
        "canonical_key": canonical,
        "raw_obj_id": str(raw_obj_id),
        "n_samples": int(len(components)),
        "components": components,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
# Known MS-HAB YCB pickable pool (from mshab/evaluate.py:88-148). Used only for
# coverage warnings -- the miner walks whatever .pkl files exist.
_KNOWN_YCB_POOL = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "013_apple",
    "024_bowl",
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--success-states-dir",
        required=True,
        help="Path to robot_success_states/ (parent of <robot_uid>/<subtask>/).",
    )
    parser.add_argument(
        "--robot", default="fetch",
        help="Robot uid subdirectory (default: fetch).",
    )
    parser.add_argument(
        "--subtask", default="pick", choices=["pick", "open", "close"],
        help="Successful manipulation subtask. Place is excluded because its "
             "success pose is the robot rest pose.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output affordances.json path.",
    )
    parser.add_argument(
        "--merge-existing", action="store_true",
        help="Preserve entities already present in --out and update/add the "
             "entities mined by this run.",
    )
    parser.add_argument(
        "--n-components", type=int, default=4,
        help="Deprecated compatibility flag. Raw success samples are emitted; "
             "use --max-samples to cap candidate count.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="Cap per-object samples after random subsampling (default 2000; "
             "0 disables capping).",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Path to fetch.urdf. Defaults to ManiSkill's packaged Fetch URDF.",
    )
    parser.add_argument(
        "--tcp-link", default="gripper_link",
        help="Link name to read as TCP (default: gripper_link).",
    )
    parser.add_argument(
        "--tcp-approach-axis", default="0,0,1",
        help="Gripper-link local approach axis as 'x,y,z' (default 0,0,1).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    approach_axis_local = np.asarray(
        [float(x) for x in args.tcp_approach_axis.split(",")], dtype=float
    )
    _setup_logging(args.verbose)

    root = Path(args.success_states_dir) / args.robot / args.subtask
    if not root.is_dir():
        log.error("expected directory %s does not exist", root)
        return 2

    # FK is only a fallback for legacy (schema < 4) PKLs that do not carry
    # ``tcp_pose_wrt_base``. Schema-v4 PKLs are anchor-correct without FK and
    # do not need a URDF, so we set ``fk`` to None when FK setup fails rather
    # than aborting.
    urdf_path = args.urdf or _default_fetch_urdf()
    fk: Optional[_FetchFK] = None
    if urdf_path:
        log.info("FK URDF (legacy fallback): %s", urdf_path)
        try:
            fk = _FetchFK(urdf_path, tcp_link_name=args.tcp_link)
        except Exception as exc:
            log.warning(
                "FK setup failed (%s); legacy PKLs without tcp_pose_wrt_base "
                "will be skipped",
                exc,
            )
    else:
        log.warning(
            "Could not locate fetch.urdf; legacy PKLs without "
            "tcp_pose_wrt_base will be skipped"
        )

    pkls = sorted(root.glob("*.pkl"))
    if not pkls:
        log.error("no .pkl files under %s", root)
        return 2

    seen_canonical: set = set()
    by_object: Dict[str, Dict] = {}
    for pkl_path in pkls:
        rec = _mine_object(
            pkl_path, fk=fk,
            max_samples=args.max_samples,
            approach_axis_local=approach_axis_local,
        )
        if rec is None:
            continue
        key = rec["canonical_key"]
        if key in seen_canonical:
            log.warning("duplicate canonical key %s from %s -- ignoring second",
                        key, pkl_path)
            continue
        seen_canonical.add(key)
        by_object[key] = {
            "raw_obj_id": rec["raw_obj_id"],
            "n_samples": rec["n_samples"],
            "components": rec["components"],
        }

    # Coverage warnings apply only to the known pickable actor pool.
    if args.subtask == "pick":
        for ycb in _KNOWN_YCB_POOL:
            if f"actor:{ycb}" not in by_object:
                log.warning("no data for canonical key %s -- runtime will emit no "
                            "affordance edges for it", ycb)

    if args.merge_existing and os.path.isfile(args.out):
        try:
            with open(args.out) as stream:
                existing = json.load(stream)
            existing_objects = existing.get("objects", {})
            if isinstance(existing_objects, dict):
                by_object = {**existing_objects, **by_object}
        except Exception as exc:
            log.error("failed to merge existing affordance asset %s: %r", args.out, exc)
            return 2

    payload = {
        "_README": (
            "anchor=[x,y,z] OBJECT frame (m); approach_dir=[x,y,z] OBJECT-frame "
            "unit approach axis (optional); width=preferred gripper qpos-sum "
            "(m). Mined by tools/build_affordances.py from "
            "robot_success_states/<robot>/<subtask>/<obj_id>.pkl. One component is "
            "stored per usable success grasp pose, keyed by canonical MS-HAB "
            "obj_id (no 'env-N_' prefix, no '-N' instance suffix)."
        ),
        "_schema_version": 2,
        "objects": by_object,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    log.info("wrote %d objects (%d components total) to %s",
             len(by_object),
             sum(len(v["components"]) for v in by_object.values()),
             out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
