"""Torch adapter over the shared graph obs builder.

SAC steps after the vector env auto-reset, so its ``done`` mask marks the first
frame of the next episode. uint16 has no torch dtype, so entity ids widen to
int32 on the way out.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Optional

import numpy as np
import torch

from teemo_sim_probe.adapters.graph_obs import (
    GraphObsBuilder as _GraphObsBuilder,
    build_graph_obs as _build_graph_obs,
)


class GraphObsBuilder(_GraphObsBuilder):

    @property
    def obs_spec_dtypes(self) -> Dict[str, np.dtype]:
        return {
            k: (np.int32 if v == np.uint16 else v)
            for k, v in super().obs_spec_dtypes.items()
        }

    def step(self, done_mask: Optional[torch.Tensor], device: torch.device):
        first = None
        if done_mask is not None:
            first = done_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        packed = super().step(is_first=first)
        return {
            k: torch.as_tensor(
                v.astype(np.int32) if v.dtype == np.uint16 else v, device=device,
            )
            for k, v in packed.items()
        }

    def reset(self, device: torch.device):
        self._frames[:] = 0
        self._last_packed = [None for _ in range(self.num_envs)]
        return self.step(
            torch.ones(self.num_envs, dtype=torch.bool), device,
        )


build_graph_obs = partial(_build_graph_obs, builder_cls=GraphObsBuilder)
