"""Network primitives for SAC.

PlainConv lifted from ManiSkill's sac_rgbd baseline; MLP and EncoderObsWrapper
adapted to the dict-obs schema we use here:

    obs is a dict that always contains 'state' and optionally:
      - 'rgb'   : uint8 [N, H, W, C_rgb]   (channels-last)
      - 'depth' : float [N, H, W, C_depth] (channels-last; pre-normalized)

Both image keys are channels-last to match how ManiSkill returns them; the
encoder permutes to channels-first before the CNN.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def make_mlp(
    in_channels: int,
    mlp_channels: List[int],
    act_builder=nn.ReLU,
    last_act: bool = True,
) -> nn.Sequential:
    c_in = in_channels
    layers: List[nn.Module] = []
    for idx, c_out in enumerate(mlp_channels):
        layers.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            layers.append(act_builder())
        c_in = c_out
    return nn.Sequential(*layers)


class PlainConv(nn.Module):
    """Simple 5-block CNN used by the ManiSkill SAC baselines.

    Accepts ``in_channels`` to handle RGB (3), depth (1), or stacked variants.
    The MaxPool kernel adapts to image_size so the final feature is always
    64 * 4 * 4 = 1024.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 256,
        image_size=(64, 64),
        last_act: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        first_pool = 4 if image_size[0] == 128 and image_size[1] == 128 else 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(first_pool, first_pool),
            nn.Conv2d(16, 32, 3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 1, padding=0),           nn.ReLU(inplace=True),
        )
        self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=last_act)
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)
        x = x.flatten(1)
        return self.fc(x)


class EncoderObsWrapper(nn.Module):
    """Wrap a CNN so it accepts the dict-obs format.

    Handles:
      - channels-last -> channels-first permute
      - uint8 RGB normalization to [0, 1]
      - optional depth concat (depth is already float [0, 1])

    The encoder is intentionally state-agnostic: callers concat `obs['state']`
    after embedding the pixels.
    """

    def __init__(self, encoder: PlainConv, rgb_key: Optional[str] = None,
                 depth_key: Optional[str] = None):
        super().__init__()
        self.encoder = encoder
        self.rgb_key = rgb_key
        self.depth_key = depth_key

    @property
    def out_dim(self) -> int:
        return self.encoder.out_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.rgb_key is not None and self.rgb_key in obs:
            rgb = obs[self.rgb_key].float() / 255.0
            parts.append(rgb)
        if self.depth_key is not None and self.depth_key in obs:
            depth = obs[self.depth_key].float()
            parts.append(depth)
        if not parts:
            raise ValueError(
                f"EncoderObsWrapper expected one of {self.rgb_key!r} / "
                f"{self.depth_key!r} in obs, got keys={list(obs.keys())}")
        img = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        img = img.permute(0, 3, 1, 2).contiguous()       # NHWC -> NCHW
        return self.encoder(img)


def infer_image_channels(sample_obs: Dict[str, torch.Tensor],
                          rgb_key: Optional[str],
                          depth_key: Optional[str]) -> int:
    """Total channels of the concatenated image input, given a sample batch."""
    c = 0
    if rgb_key is not None and rgb_key in sample_obs:
        c += sample_obs[rgb_key].shape[-1]
    if depth_key is not None and depth_key in sample_obs:
        c += sample_obs[depth_key].shape[-1]
    return c


def infer_image_size(sample_obs: Dict[str, torch.Tensor],
                      rgb_key: Optional[str],
                      depth_key: Optional[str]):
    for k in (rgb_key, depth_key):
        if k is not None and k in sample_obs:
            return tuple(sample_obs[k].shape[1:3])
    raise ValueError("No image key found to infer image size")
