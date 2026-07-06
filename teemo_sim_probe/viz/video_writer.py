"""Optional: stitch saved overlay + graph PNG pairs into an mp4.

Pure-matplotlib frame compositing + imageio for encoding (both common deps).
If imageio is unavailable this degrades to writing a contact-sheet PNG.
"""

from __future__ import annotations

from typing import List

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
    try:
        import imageio.v2 as imageio
        # Pad all frames to a common size that is divisible by 16 so ffmpeg
        # does not silently resize the video for codec compatibility.
        H = W = 0
        for op, gp in zip(overlay_paths, graph_paths):
            frame = _hstack(_load_png(op), _load_png(gp))
            H = max(H, frame.shape[0])
            W = max(W, frame.shape[1])
        H = _round_up(H, 16)
        W = _round_up(W, 16)

        with imageio.get_writer(out_path, fps=fps) as writer:
            for op, gp in zip(overlay_paths, graph_paths):
                frame = _hstack(_load_png(op), _load_png(gp))
                out = np.full((H, W, 3), 253, dtype=np.uint8)
                out[: frame.shape[0], : frame.shape[1]] = frame
                writer.append_data(out)
        return out_path
    except Exception:
        # Fallback contact sheet without materializing the full eval rollout.
        frames = []
        for op, gp in list(zip(overlay_paths, graph_paths))[:8]:
            frames.append(_hstack(_load_png(op), _load_png(gp)))
        if not frames:
            raise
        sheet = np.concatenate(frames, axis=0)
        import matplotlib.image as mpimg
        png = out_path.rsplit(".", 1)[0] + "_contactsheet.png"
        mpimg.imsave(png, sheet)
        return png


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
