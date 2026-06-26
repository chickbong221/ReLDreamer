"""One-shot diagnostic: re-derive anchors from the saved success pickles and
print summary stats. Helps decide whether the mined anchors are oversized
because ``obj_pose_wrt_base`` is wrong, or because the math is wrong.

Usage:
    python -m teemo_sim_probe.tools.inspect_affordance_anchors
    python -m teemo_sim_probe.tools.inspect_affordance_anchors --root /path/to/robot_success_states/fetch/pick
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np


def quat_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def default_root() -> str:
    r = os.path.expandvars("$MS_ASSET_DIR/data/robot_success_states/fetch/pick")
    if os.path.isdir(r):
        return r
    return os.path.expanduser("~/.maniskill/data/robot_success_states/fetch/pick")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None,
                        help="Directory containing <obj>.pkl files. "
                             "Default: $MS_ASSET_DIR/.../fetch/pick.")
    args = parser.parse_args()

    root = args.root or default_root()
    if not os.path.isdir(root):
        print(f"ERROR: directory not found: {root}")
        return 2

    pkls = sorted(glob.glob(os.path.join(root, "*.pkl")))
    if not pkls:
        print(f"ERROR: no .pkl files under {root}")
        return 2

    print(f"root: {root}")
    print(f"{'file':35s} {'n':>3s}  "
          f"{'obj_p_mean':>26s}  {'obj_q_mean':>34s}  "
          f"{'|a|_mean':>9s} {'|a|_min':>8s} {'|a|_max':>8s}")
    for p in pkls:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as exc:
            print(f"{os.path.basename(p):35s} [load failed: {exc!r}]")
            continue

        obj = np.asarray(d.get("obj_pose_wrt_base", []), dtype=float)
        tcp = np.asarray(d.get("tcp_pose_wrt_base", []), dtype=float)
        if obj.ndim != 2 or tcp.ndim != 2 or obj.shape[0] == 0:
            print(f"{os.path.basename(p):35s} [no pose arrays]")
            continue

        obj_p, obj_q, tcp_p = obj[:, :3], obj[:, 3:7], tcp[:, :3]
        anchors = np.stack([
            quat_R(q).T @ (t - o) for o, q, t in zip(obj_p, obj_q, tcp_p)
        ])
        mag = np.linalg.norm(anchors, axis=1)
        print(
            f"{os.path.basename(p):35s} {len(obj):3d}  "
            f"{str(obj_p.mean(0).round(3)):>26s}  "
            f"{str(obj_q.mean(0).round(3)):>34s}  "
            f"{mag.mean():9.3f} {mag.min():8.3f} {mag.max():8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
