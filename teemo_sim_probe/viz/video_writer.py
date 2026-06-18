"""Optional: stitch saved overlay + graph PNG pairs into an mp4.

Pure-matplotlib frame compositing + imageio for encoding (both common deps).
If imageio is unavailable this degrades to writing a contact-sheet PNG.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np


def _load_png(path: str) -> np.ndarray:
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _hstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])

    def pad(x):
        if x.shape[0] == h:
            return x
        out = np.full((h, x.shape[1], 3), 253, dtype=np.uint8)
        out[: x.shape[0]] = x
        return out

    a, b = pad(a), pad(b)
    return np.concatenate([a, b], axis=1)


def write_video(
    overlay_paths: List[str],
    graph_paths: List[str],
    out_path: str,
    fps: int = 5,
) -> str:
    frames = []
    for op, gp in zip(overlay_paths, graph_paths):
        frames.append(_hstack(_load_png(op), _load_png(gp)))

    try:
        import imageio.v2 as imageio
        # Pad all frames to a common size.
        H = max(f.shape[0] for f in frames)
        W = max(f.shape[1] for f in frames)
        padded = []
        for f in frames:
            out = np.full((H, W, 3), 253, dtype=np.uint8)
            out[: f.shape[0], : f.shape[1]] = f
            padded.append(out)
        imageio.mimsave(out_path, padded, fps=fps)
        return out_path
    except Exception:
        # Fallback contact sheet.
        sheet = np.concatenate(frames[:8], axis=0)
        import matplotlib.image as mpimg
        png = out_path.rsplit(".", 1)[0] + "_contactsheet.png"
        mpimg.imsave(png, sheet)
        return png
