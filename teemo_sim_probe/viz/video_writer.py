"""Optional: stitch N per-frame PNG panels into an mp4.

Pure-matplotlib frame compositing + imageio for encoding (both common deps).
If imageio is unavailable this degrades to writing a contact-sheet PNG.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _load_png(path: str) -> np.ndarray:
    import matplotlib.image as mpimg
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _hstack(panels: Sequence[np.ndarray]) -> np.ndarray:
    h = max(p.shape[0] for p in panels)

    def pad(x):
        if x.shape[0] == h:
            return x
        out = np.full((h, x.shape[1], 3), 253, dtype=np.uint8)
        out[: x.shape[0]] = x
        return out

    return np.concatenate([pad(p) for p in panels], axis=1)


def write_video(
    panel_lists: Sequence[Sequence[str]],
    out_path: str,
    fps: int = 5,
) -> str:
    """Encode a video from horizontally-stacked panels.

    ``panel_lists[i]`` is a list of PNG paths for panel ``i`` (one entry per
    frame). All panel lists must have the same length; frame ``t`` is the
    hstack of ``panel_lists[0][t], panel_lists[1][t], ...``.
    """
    if not panel_lists:
        raise ValueError("write_video: panel_lists is empty")
    n_frames = len(panel_lists[0])
    if any(len(p) != n_frames for p in panel_lists):
        raise ValueError("write_video: panel lists must be same length")
    if n_frames == 0:
        raise ValueError("write_video: no frames")

    try:
        import imageio.v2 as imageio
        # Pad all frames to a common size that is divisible by 16 so ffmpeg
        # does not silently resize the video for codec compatibility.
        H = W = 0
        for t in range(n_frames):
            frame = _hstack([_load_png(p[t]) for p in panel_lists])
            H = max(H, frame.shape[0])
            W = max(W, frame.shape[1])
        H = _round_up(H, 16)
        W = _round_up(W, 16)

        with imageio.get_writer(out_path, fps=fps) as writer:
            for t in range(n_frames):
                frame = _hstack([_load_png(p[t]) for p in panel_lists])
                out = np.full((H, W, 3), 253, dtype=np.uint8)
                out[: frame.shape[0], : frame.shape[1]] = frame
                writer.append_data(out)
        return out_path
    except Exception:
        # Fallback contact sheet without materializing the full eval rollout.
        frames: List[np.ndarray] = []
        for t in range(min(n_frames, 8)):
            frames.append(_hstack([_load_png(p[t]) for p in panel_lists]))
        if not frames:
            raise
        sheet = np.concatenate(frames, axis=0)
        import matplotlib.image as mpimg
        png = out_path.rsplit(".", 1)[0] + "_contactsheet.png"
        mpimg.imsave(png, sheet)
        return png


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
