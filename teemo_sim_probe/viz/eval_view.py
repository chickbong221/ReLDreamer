"""Third-person evaluation-camera render + success_once line plot.

eval camera : env.render() with render_mode="all" returns a [N,H,W,3] human
              render image (e.g. 512x768 for Fetch).
success_once: MS-HAB info has per-step ``success`` but no ``success_once``;
              success_once[t] = max(success[0..t]) (did it ever succeed by t).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def render_eval_view(env, env_idx: int = 0) -> Optional[np.ndarray]:
    """Return a [H,W,3] uint8 third-person frame from env.render(), or None."""
    try:
        img = env.unwrapped.render()
    except Exception:
        try:
            img = env.render()
        except Exception:
            return None
    img = _to_np(img)
    if img.ndim == 4:
        img = img[env_idx]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_image(img: np.ndarray, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    mpimg.imsave(path, img)
    return path


class SuccessTracker:
    """Records per-step success and the running success_once flag."""

    def __init__(self, env_idx: int = 0):
        self.env_idx = env_idx
        self.steps: List[int] = []
        self.success: List[int] = []
        self.success_once: List[int] = []
        self._ever = 0

    def update(self, info: dict, frame: int) -> None:
        s = _success_from_info(info, self.env_idx)
        self._ever = max(self._ever, s)
        self.steps.append(frame)
        self.success.append(s)
        self.success_once.append(self._ever)

    def save_plot(self, path: str, title: str = "success_once") -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=130)
        ax.plot(self.steps, self.success_once, color="#2a9d4a",
                lw=2.0, label="success_once")
        ax.plot(self.steps, self.success, color="#bbbbbb", lw=1.0,
                ls="--", alpha=0.8, label="success (per-step)")
        ax.set_xlabel("step")
        ax.set_ylabel("success")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def save_csv(self, path: str) -> str:
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "success", "success_once"])
            for s, su, so in zip(self.steps, self.success, self.success_once):
                w.writerow([s, su, so])
        return path


def _success_from_info(info: dict, env_idx: int) -> int:
    if "success" in info:
        return int(bool(_to_np(info["success"]).reshape(-1)[env_idx]))

    final_info = info.get("final_info", {})
    episode = final_info.get("episode", {}) if isinstance(final_info, dict) else {}
    for key in ("success_once", "success_at_end", "success"):
        if key in episode:
            return int(bool(_to_np(episode[key]).reshape(-1)[env_idx]))

    episode = info.get("episode", {})
    if isinstance(episode, dict):
        for key in ("s_o", "s_e", "success_once", "success_at_end", "success"):
            if key in episode:
                return int(bool(_to_np(episode[key]).reshape(-1)[env_idx]))
    return 0
