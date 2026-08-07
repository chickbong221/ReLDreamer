"""Frozen appearance encoder: batching, token selection, and pooling.

The real checkpoint is never loaded here. What matters is the plumbing around
it -- that both cameras ride one forward, that only patch tokens survive, and
that a node with no pixels in a camera pools to exactly zero rather than to a
normalised ratio of noise.
"""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the sim tests
    torch = None

if torch is not None:
    from scenegraph.adapters.dino import _DIMS, DinoFeatures

RES, GRID, PATCHES, DIM = 112, 8, 64, 384
REGISTERS = 4


class _StubViT(torch.nn.Module if torch is not None else object):
    """Returns class, register and patch tokens the way DINOv2 does.

    ``x_norm_patchtokens`` is already register-free upstream; returning the
    other entries alongside it is what makes the assertion meaningful.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen = []

    def forward_features(self, x):
        self.seen.append(tuple(x.shape))
        B = x.shape[0]
        # Patch r is ones with dimension r raised: patches differ in direction,
        # not only in magnitude, which L2 normalisation would erase.
        patches = torch.ones(PATCHES, DIM)
        patches[torch.arange(PATCHES), torch.arange(PATCHES)] = 2.0
        return {
            'x_norm_clstoken': torch.full((B, DIM), -1.0),
            'x_norm_regtokens': torch.full((B, REGISTERS, DIM), -2.0),
            'x_norm_patchtokens': patches[None].expand(B, PATCHES, DIM).clone(),
        }


def _features():
    """A DinoFeatures with the hub load skipped."""
    obj = object.__new__(DinoFeatures)
    obj.name = 'dinov2_vits14_reg'
    obj.res, obj.dim = RES, DIM
    obj.grid, obj.patches = GRID, PATCHES
    obj.device = torch.device('cpu')
    obj.checksum = 'stub'
    obj._model = _StubViT()
    shape = lambda v: torch.tensor(v).view(1, 3, 1, 1)
    obj._mean = shape((0.485, 0.456, 0.406))
    obj._std = shape((0.229, 0.224, 0.225))
    return obj


@unittest.skipIf(torch is None, 'torch is not installed')
class ModelRegistryTests(unittest.TestCase):

    def test_register_variants_are_known(self):
        self.assertEqual(_DIMS['dinov2_vits14_reg'], 384)
        self.assertEqual(_DIMS['dinov2_vitb14_reg'], 768)

    def test_resolution_must_tile_the_patch_size(self):
        with self.assertRaises(ValueError):
            DinoFeatures('dinov2_vits14_reg', res=100)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            DinoFeatures('resnet50')


@unittest.skipIf(torch is None, 'torch is not installed')
class BatchingTests(unittest.TestCase):

    def setUp(self):
        self.dino = _features()
        self.rgb = torch.randint(0, 256, (3, 2, RES, RES, 3), dtype=torch.uint8)

    def test_both_cameras_ride_one_forward(self):
        tokens = self.dino.patch_tokens(self.rgb)
        self.assertEqual(tuple(tokens.shape), (3, 2, PATCHES, DIM))
        self.assertEqual(len(self.dino._model.seen), 1)
        self.assertEqual(self.dino._model.seen[0], (6, 3, RES, RES))

    def test_class_and_register_tokens_are_dropped(self):
        tokens = self.dino.patch_tokens(self.rgb)
        self.assertGreaterEqual(float(tokens.min()), 0.0)


@unittest.skipIf(torch is None, 'torch is not installed')
class PoolingTests(unittest.TestCase):

    def setUp(self):
        self.dino = _features()
        self.tokens = self.dino.patch_tokens(
            torch.zeros((1, 2, RES, RES, 3), dtype=torch.uint8))

    def _pool(self, weights):
        return self.dino.pool(self.tokens, weights)

    def test_shape_keeps_the_camera_axis(self):
        w = torch.zeros((1, 2, 3, PATCHES))
        self.assertEqual(tuple(self._pool(w).shape), (1, 2, 3, DIM))

    def test_empty_support_pools_to_exactly_zero(self):
        w = torch.zeros((1, 2, 1, PATCHES))
        self.assertEqual(float(self._pool(w).abs().sum()), 0.0)

    def test_nonempty_support_is_l2_normalised(self):
        w = torch.zeros((1, 2, 1, PATCHES))
        w[0, 0, 0, 5] = 1.0
        out = self._pool(w)
        self.assertAlmostEqual(float(out[0, 0, 0].norm()), 1.0, places=5)
        self.assertEqual(float(out[0, 1, 0].abs().sum()), 0.0)

    def test_uniform_scaling_of_one_node_cancels(self):
        a = torch.zeros((1, 2, 1, PATCHES))
        a[0, 0, 0, :4] = 1.0
        b = a * 0.25
        torch.testing.assert_close(self._pool(a), self._pool(b))

    def test_cameras_pool_independently(self):
        w = torch.zeros((1, 2, 1, PATCHES))
        w[0, 0, 0, 1] = 1.0
        w[0, 1, 0, 60] = 1.0
        out = self._pool(w)
        self.assertGreater(float((out[0, 0, 0] - out[0, 1, 0]).abs().max()), 0)


if __name__ == '__main__':
    unittest.main()
