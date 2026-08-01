"""Tests for checkpoint-driven raw RGB export mode."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

from teemo_sim_probe.core.mask_extractor import read_unwrapped_rgbs
from teemo_sim_probe.run_mshab_probe import (
    _export_rgb_rollout,
    _safe_path_component,
    _save_rgb_png,
    parse_args,
)


class _FakeEnv:
    def __init__(self):
        self.step_count = 0

    @property
    def unwrapped(self):
        return self

    def get_obs(self):
        return {
            "sensor_data": {
                "fetch_head": {
                    "rgb": np.full((1, 2, 3, 3), 260.0, dtype=np.float32),
                    "depth": np.zeros((1, 2, 3, 1), dtype=np.float32),
                },
                "fetch_hand": {
                    "rgb": np.zeros((1, 2, 3, 3), dtype=np.uint8),
                },
            }
        }

    def step(self, _action):
        self.step_count += 1
        return {}, 0.0, False, False, {}


class _FakePolicy:
    def act(self, _obs):
        return np.zeros(1, dtype=np.float32)


class TestRgbExport(unittest.TestCase):
    def test_rgb_only_defaults_to_200_frames(self):
        self.assertEqual(parse_args(["--rgb-only"]).steps, 200)
        self.assertEqual(parse_args([]).steps, 60)

    def test_explicit_step_count_is_preserved(self):
        self.assertEqual(parse_args(["--rgb-only", "--steps", "7"]).steps, 7)

    def test_reads_all_rgb_cameras_without_segmentation(self):
        frames = read_unwrapped_rgbs(_FakeEnv())
        self.assertEqual(list(frames), ["fetch_head", "fetch_hand"])
        self.assertEqual(frames["fetch_head"].shape, (2, 3, 3))
        self.assertEqual(frames["fetch_head"].dtype, np.uint8)
        self.assertTrue(np.all(frames["fetch_head"] == 255))

    def test_camera_uid_is_safe_as_a_directory(self):
        self.assertEqual(_safe_path_component("wrist/camera 1"), "wrist_camera_1")

    def test_png_is_saved_as_truecolor_rgb(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            _save_rgb_png(np.zeros((2, 3, 3), dtype=np.uint8), str(path))
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (3, 2))

    def test_rollout_writes_requested_count_for_every_camera(self):
        with TemporaryDirectory() as tmp:
            env = _FakeEnv()
            args = SimpleNamespace(steps=3, out=tmp)
            _export_rgb_rollout(env, _FakePolicy(), {}, args)

            for camera in ("fetch_head", "fetch_hand"):
                frames = sorted((Path(tmp) / camera).glob("*.png"))
                self.assertEqual(len(frames), 3)
                self.assertEqual(frames[-1].name, "frame_0002.png")


if __name__ == "__main__":
    unittest.main()
