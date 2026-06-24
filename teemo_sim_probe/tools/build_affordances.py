"""Offline miner: MS-HAB pick-success rollouts -> ``affordances.json``.

Walks ``ASSET_DIR/robot_success_states/<robot_uid>/pick/<obj_id>.pkl`` (saved
by ``mshab.envs.wrappers.collect_data.FetchCollectRobotInitWrapper``) and
produces a sparse set of affordance components per object:

    component_k = {
        anchor_obj_frame: 3D point in OBJECT frame (pose-invariant),
        preferred_width: gripper qpos-sum at success (matches runtime),
    }

Per success sample:

    width = robot_qpos[-2] + robot_qpos[-1]    # qpos-sum convention
    tcp_in_base = FK(Fetch, robot_qpos)        # SAPIEN required
    anchor_obj  = inv(obj_pose_wrt_base) * tcp_in_base   (position only)

N samples per object -> K components via farthest-point + k-means on anchors;
preferred_width = cluster median.

PICK ONLY. Place success is invalid for affordance mining: the place check
requires ``~is_grasped`` with TCP at ``ee_rest_world_pose``
(mshab/envs/sequential_task.py:1138), so ``inv(obj) * tcp`` would learn the
robot's rest pose rather than an object affordance.

If FK is unavailable (no SAPIEN / no Fetch URDF), the object is skipped with
a clear error -- we deliberately do NOT write placeholder anchors, because
``[0,0,0]`` would produce valid-looking but false runtime relations.

Usage::

    python -m teemo_sim_probe.tools.build_affordances \\
        --success-states-dir $MS_ASSET_DIR/data/robot_success_states \\
        --out teemo_sim_probe/configs/affordances.json \\
        --robot fetch \\
        --n-components 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
log = logging.getLogger("build_affordances")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# Canonical key (must match runtime exactly)
# --------------------------------------------------------------------------- #
def _canonical_key(name: Optional[str]) -> Optional[str]:
    """Mirror of teemo_sim_probe.core.affordance.canonical_affordance_key.

    Kept inlined so the miner has no runtime-package dependency beyond what
    the importer provides. We deliberately import the runtime helper here so
    the two implementations cannot drift.
    """
    from teemo_sim_probe.core.affordance import canonical_affordance_key
    return canonical_affordance_key(name)


# --------------------------------------------------------------------------- #
# Pose math (positions only -- we don't need orientation for the anchor)
# --------------------------------------------------------------------------- #
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
    """Apply ``inv(pose) * point`` where pose = [x,y,z,qw,qx,qy,qz]."""
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
    """Rotate a world-frame direction into the OBJECT frame: ``R_obj.T @ d``."""
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


# --------------------------------------------------------------------------- #
# FK (SAPIEN-based, Fetch only)
# --------------------------------------------------------------------------- #
class _FetchFK:
    """Stateful FK helper. Loads the Fetch URDF once, then per-sample sets
    qpos and reads the gripper_link world pose (== base frame because the
    root is fixed at identity).
    """

    def __init__(self, urdf_path: str, tcp_link_name: str = "gripper_link"):
        # Imports are local so the miner can fail clearly with --help even
        # when SAPIEN / ManiSkill aren't installed.
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
        # SAPIEN's URDF loader attaches visual shapes; without a render
        # system in the scene, loader.load() raises
        # "no system with name [render] is added to scene" and the scene's
        # __del__ then aborts with the same error (see
        # ManiSkill/mani_skill/envs/sapien_env.py:1213-1217 -- ManiSkill
        # always adds a RenderSystem alongside PhysX for the same reason).
        # We never actually render here; the system just needs to exist.
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
            # Some SAPIEN versions don't expose set_root_pose on articulations;
            # fix_root_link=True already pins it at identity by default.
            pass

        # Resolve the gripper link.
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

        # SAPIEN articulation qpos length may differ from MS-HAB's saved qpos
        # if some joints are fixed/mimic in the load. Warn loudly.
        try:
            self._qpos_dim = int(self._robot.dof)
        except Exception:
            self._qpos_dim = -1

    def tcp_in_base(self, qpos: np.ndarray) -> Optional[np.ndarray]:
        """Return TCP pose [x,y,z,qw,qx,qy,qz] in base (== world) frame."""
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
    """Find ``fetch.urdf`` via ManiSkill's PACKAGE_ASSET_DIR (best effort)."""
    try:
        from mani_skill import PACKAGE_ASSET_DIR
    except Exception:
        return None
    return os.path.join(PACKAGE_ASSET_DIR, "robots", "fetch", "fetch.urdf")


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _farthest_point_init(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(points)
    if n <= k:
        return points.copy()
    idx0 = int(rng.integers(0, n))
    centers = [points[idx0]]
    dists = np.linalg.norm(points - points[idx0], axis=1)
    for _ in range(k - 1):
        next_idx = int(np.argmax(dists))
        centers.append(points[next_idx])
        new_dists = np.linalg.norm(points - points[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return np.stack(centers, axis=0)


def _kmeans(points: np.ndarray, k: int, max_iter: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with farthest-point init. Returns (centers, labels)."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n <= k:
        labels = np.arange(n)
        return points.copy(), labels

    centers = _farthest_point_init(points, k, rng)
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        diffs = points[:, None, :] - centers[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
        new_labels = np.argmin(d2, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = points[mask].mean(axis=0)
            else:
                # Reseed empty cluster with the farthest point from any center.
                diffs = points[:, None, :] - centers[None, :, :]
                d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
                far_idx = int(np.argmax(d2.min(axis=1)))
                centers[j] = points[far_idx]
    return centers, labels


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
    required = ("obj_id", "robot_qpos", "obj_pose_wrt_base")
    if not all(r in data for r in required):
        log.error("missing keys in %s: have=%s, need=%s",
                  path, sorted(data), required)
        return None
    return data


def _mine_object(
    pkl_path: Path, fk: _FetchFK, n_components: int, max_samples: int,
    approach_axis_local: np.ndarray,
) -> Optional[Dict]:
    data = _load_pkl(pkl_path)
    if data is None:
        return None

    qpos_list = np.asarray(data["robot_qpos"], dtype=float)
    pose_list = np.asarray(data["obj_pose_wrt_base"], dtype=float)
    raw_obj_id = data.get("obj_id")
    canonical = _canonical_key(raw_obj_id)
    if canonical is None:
        log.error("%s: obj_id %r did not canonicalize", pkl_path, raw_obj_id)
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

    n_samples = qpos_list.shape[0]
    if n_samples == 0:
        log.warning("%s: 0 samples, skipping", pkl_path)
        return None

    # Subsample if too many.
    if 0 < max_samples < n_samples:
        sel = np.random.default_rng(0).choice(n_samples, size=max_samples, replace=False)
        qpos_list = qpos_list[sel]
        pose_list = pose_list[sel]
        n_samples = max_samples

    anchors_obj: List[np.ndarray] = []
    widths: List[float] = []
    approaches_obj: List[Optional[np.ndarray]] = []
    fk_failures = 0
    for i in range(n_samples):
        qpos = qpos_list[i]
        obj_pose = pose_list[i, :7]
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
        log.error("%s: no usable samples after FK -- skipping", pkl_path)
        return None

    anchors_arr = np.stack(anchors_obj, axis=0)        # (N, 3)
    widths_arr = np.asarray(widths, dtype=float)        # (N,)
    k = min(n_components, len(anchors_arr))
    centers, labels = _kmeans(anchors_arr, k=k, seed=0)

    components: List[Dict] = []
    for j in range(k):
        mask = labels == j
        if not mask.any():
            continue
        # Use cluster MEDIAN for both anchor and width to be robust to outliers.
        anchor_med = np.median(anchors_arr[mask], axis=0).tolist()
        width_med = float(np.median(widths_arr[mask]))
        # Median approach direction over the cluster's valid samples (then
        # renormalize). If no sample in the cluster had a direction, omit it.
        comp_dirs = [approaches_obj[idx] for idx in np.where(mask)[0]
                     if approaches_obj[idx] is not None]
        approach_field: Dict[str, List[float]] = {}
        if comp_dirs:
            d_med = np.median(np.stack(comp_dirs, axis=0), axis=0)
            n = float(np.linalg.norm(d_med))
            if n > 1e-9:
                approach_field = {
                    "approach_dir": [round(float(x), 6) for x in (d_med / n)]
                }
        components.append({
            "anchor": [round(float(x), 6) for x in anchor_med],
            **approach_field,
            "width": round(width_med, 6),
            "n_support": int(mask.sum()),
        })

    log.info("%s -> %s : %d components from %d samples",
             pkl_path.name, canonical, len(components), len(anchors_arr))
    return {
        "canonical_key": canonical,
        "raw_obj_id": str(raw_obj_id),
        "n_samples": int(len(anchors_arr)),
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
        help="Path to robot_success_states/ (parent of <robot_uid>/pick/).",
    )
    parser.add_argument(
        "--robot", default="fetch",
        help="Robot uid subdirectory (default: fetch).",
    )
    parser.add_argument(
        "--subtask", default="pick", choices=["pick"],
        help="Only 'pick' is supported (place success has ~is_grasped).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output affordances.json path.",
    )
    parser.add_argument(
        "--n-components", type=int, default=4,
        help="K components per object (default 4).",
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

    urdf_path = args.urdf or _default_fetch_urdf()
    if not urdf_path:
        log.error("Could not locate fetch.urdf. Pass --urdf explicitly.")
        return 2
    log.info("FK URDF: %s", urdf_path)

    try:
        fk = _FetchFK(urdf_path, tcp_link_name=args.tcp_link)
    except Exception as exc:
        log.error("FK setup failed: %s", exc)
        log.error("Refusing to write placeholder anchors; aborting.")
        return 2

    pkls = sorted(root.glob("*.pkl"))
    if not pkls:
        log.error("no .pkl files under %s", root)
        return 2

    seen_canonical: set = set()
    by_object: Dict[str, Dict] = {}
    for pkl_path in pkls:
        rec = _mine_object(
            pkl_path, fk=fk,
            n_components=args.n_components,
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

    # Coverage warnings against the known YCB pool.
    for ycb in _KNOWN_YCB_POOL:
        if ycb not in by_object:
            log.warning("no data for canonical key %s -- runtime will emit no "
                        "affordance edges for it", ycb)

    payload = {
        "_README": (
            "anchor=[x,y,z] OBJECT frame (m); approach_dir=[x,y,z] OBJECT-frame "
            "unit approach axis (optional); width=preferred gripper qpos-sum "
            "(m). Mined by tools/build_affordances.py from "
            "robot_success_states/<robot>/pick/<obj_id>.pkl. Keyed by canonical "
            "MS-HAB obj_id (no 'env-N_' prefix, no '-N' instance suffix)."
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
