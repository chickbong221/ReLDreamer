"""Per-camera appearance caching and the terminal re-emit.

A node that leaves one camera's view keeps that camera's last embedding and
nothing else: the other camera keeps updating, the box goes to zero, and a
camera that never saw the node stays at zero so the decoder can mask it out.

The simulator is stubbed. What is exercised is the bookkeeping around it, which
is where every ordering mistake lives.
"""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for the sim tests
    torch = None

from scenegraph.adapters.graph_obs import GraphObsBuilder
from scenegraph.adapters.graph_vocab import (
    EntityVocab, GraphVocab, build_absolute_vocab, build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.schema import Graph, Node

CAMS, DIM, GRID = 2, 4, 2
PATCHES = GRID * GRID


class _StubDino:
    """Pools each node's weights into a constant vector per camera.

    Camera ``c`` returns ``value[c]`` scaled by the node's row so a cache hit
    is distinguishable from a fresh pool, and returns exactly zero for empty
    support the way the real pooler does.
    """

    dim, grid, device = DIM, GRID, 'cpu'

    def __init__(self):
        self.value = np.array([1.0, 2.0])
        self.calls = 0
        self.shapes = []

    def patch_tokens(self, rgb):
        self.calls += 1
        return rgb

    def pool(self, tokens, weights):
        w = np.asarray(weights)
        self.shapes.append(tuple(w.shape))
        support = w.sum(-1)                       # [A, C, N]
        out = np.zeros((*support.shape, DIM), np.float32)
        for c in range(support.shape[1]):
            live = support[:, c] > 0
            out[:, c][live] = self.value[c]
        return torch.as_tensor(out) if torch is not None else out


class _Builder(GraphObsBuilder):
    """GraphObsBuilder with every simulator touch replaced."""

    def _read_segmentation(self):
        return {cam: [None] * self.num_envs for cam in self.cameras}

    def _read_rgb(self):
        return np.zeros((self.num_envs, self.n_cams, 1))

    def _purge_caches(self):
        pass

    def _refresh_scene_caches_if_needed(self):
        pass

    def _build_one(self, env_idx, episode_boundary, seg_by_cam):
        return self.plan[env_idx].pop(0)


def _node(node_id, seen):
    """One node whose per-camera patch weights say which cameras see it."""
    weights = np.zeros((CAMS, PATCHES), np.float32)
    for c in seen:
        weights[c, 0] = 1.0
    return Node(
        node_id=node_id, node_type="object", name=node_id,
        visible=bool(seen), patch_weights=weights,
        attributes={"whitelist_key": "actor:bowl"},
    )


def _graph(*nodes):
    return Graph(frame=0, env_id="env0", camera="fetch_head", nodes=list(nodes))


def _vocab():
    relation = build_relation_vocab()
    return GraphVocab(
        entity=EntityVocab({"<pad>": 0, "<ee>": 1, "actor:bowl": 2}),
        relation=relation,
        absolute=build_absolute_vocab(),
        temporal=build_temporal_vocab(),
        abs_valid=np.zeros((len(relation), 1), bool),
        temp_valid=np.zeros((len(relation),), bool),
    )


def _builder(num_envs=1):
    obj = object.__new__(_Builder)
    obj.num_envs = num_envs
    obj.vocab = _vocab()
    obj.cameras = ["fetch_head", "fetch_hand"]
    obj.record_camera = "fetch_head"
    obj.dino = _StubDino()
    obj.app_dim, obj.patch_grid = DIM, GRID
    obj.n_max, obj.e_max = 4, 8
    obj.bypass_teemo = False
    obj.env = SimpleNamespace(unwrapped=SimpleNamespace(scene=None))
    obj._appearance = [{} for _ in range(num_envs)]
    obj._last_packed = [None for _ in range(num_envs)]
    obj._fact_drops = np.zeros(num_envs, np.float32)
    obj.plan = [[] for _ in range(num_envs)]
    return obj


class SensorSourceTests(unittest.TestCase):
    """Segmentation and RGB come from the wrapper's stash, not from the env.

    MS-HAB's ``BaseEnv._last_obs`` holds only the state half, so reading it
    yields a graph built from no pixels rather than a loud failure.
    """

    def _builder(self, raw_obs):
        obj = object.__new__(GraphObsBuilder)
        obj.sensor_source = SimpleNamespace(raw_obs=raw_obs)
        obj.cameras = ['fetch_head']
        obj._cams_checked = False
        return obj

    def test_no_stash_names_the_wrapper(self):
        with self.assertRaisesRegex(RuntimeError, 'NamedCameraRGBWrapper'):
            self._builder(None)._sensor_data()

    def test_an_unwired_source_is_distinguishable_from_an_empty_stash(self):
        builder = self._builder(None)
        builder.sensor_source = None
        with self.assertRaisesRegex(RuntimeError, 'source=NoneType'):
            builder._sensor_data()

    def test_a_state_only_observation_is_rejected(self):
        builder = self._builder({'agent': {}, 'extra': {}})
        with self.assertRaisesRegex(RuntimeError, 'NamedCameraRGBWrapper'):
            builder._sensor_data()

    def test_the_stashed_sensor_data_is_returned(self):
        data = {'fetch_head': {'rgb': object(), 'segmentation': object()}}
        builder = self._builder({'sensor_data': data})
        self.assertIs(builder._sensor_data(), data)

    def test_a_camera_the_obs_mode_omits_is_named(self):
        builder = self._builder({'sensor_data': {'fetch_hand': {}}})
        with self.assertRaisesRegex(KeyError, 'fetch_head'):
            builder._sensor_data()


@unittest.skipIf(torch is None, 'torch is not installed')
class AppearanceCacheTests(unittest.TestCase):

    def _pool(self, builder, graphs):
        active = sorted(graphs)
        builder._pool_appearance(active, graphs)

    def test_a_camera_that_sees_the_node_writes_its_row(self):
        b = _builder()
        node = _node("a", seen=(0, 1))
        self._pool(b, {0: _graph(node)})
        np.testing.assert_allclose(node.appearance, [[1.0] * DIM, [2.0] * DIM])

    def test_a_camera_that_never_saw_the_node_stays_zero(self):
        b = _builder()
        node = _node("a", seen=(0,))
        self._pool(b, {0: _graph(node)})
        np.testing.assert_allclose(node.appearance[0], [1.0] * DIM)
        self.assertEqual(float(np.abs(node.appearance[1]).sum()), 0.0)

    def test_losing_one_camera_keeps_that_row_and_updates_the_other(self):
        b = _builder()
        self._pool(b, {0: _graph(_node("a", seen=(0, 1)))})
        b.dino.value = np.array([9.0, 9.0])
        later = _node("a", seen=(1,))
        self._pool(b, {0: _graph(later)})
        np.testing.assert_allclose(later.appearance[0], [1.0] * DIM)
        np.testing.assert_allclose(later.appearance[1], [9.0] * DIM)

    def test_caches_do_not_leak_between_environments(self):
        b = _builder(num_envs=2)
        self._pool(b, {0: _graph(_node("a", seen=(0, 1)))})
        blind = _node("a", seen=())
        self._pool(b, {1: _graph(blind)})
        self.assertEqual(float(np.abs(blind.appearance).sum()), 0.0)

    def test_a_retained_node_gets_a_zero_box(self):
        b = _builder()
        node = _node("a", seen=())
        node.bbox = None
        self._pool(b, {0: _graph(node)})
        self.assertEqual(node.bbox.shape, (CAMS, 4))
        self.assertTrue((node.bbox == 0).all())

    def test_a_node_pools_into_its_own_environment_row(self):
        # Buffers are indexed by env, not by position in the active list, so
        # an env that pools alone still reads back its own row.
        b = _builder(num_envs=2)
        node = _node("a", seen=(0,))
        self._pool(b, {1: _graph(node)})
        np.testing.assert_allclose(node.appearance[0], [1.0] * DIM)

    def test_the_pooling_shape_never_moves(self):
        # A buffer cut to the active set and the widest graph hands the caching
        # allocator a fresh block size per combination, which it keeps.
        b = _builder(num_envs=3)
        self._pool(b, {0: _graph(_node("a", seen=(0,)))})
        self._pool(b, {
            1: _graph(_node("b", seen=(0,)), _node("c", seen=(1,))),
            2: _graph(_node("d", seen=(0, 1))),
        })
        self.assertEqual(set(b.dino.shapes), {(3, CAMS, b.n_max, PATCHES)})


@unittest.skipIf(torch is None, 'torch is not installed')
class TerminalGuardTests(unittest.TestCase):

    def _step(self, b, **kw):
        return b.step(**kw)

    def test_is_first_clears_only_that_environment(self):
        b = _builder(num_envs=2)
        b._appearance[0]["a"] = np.ones((CAMS, DIM), np.float32)
        b._appearance[1]["a"] = np.ones((CAMS, DIM), np.float32)
        b.plan = [[_graph(_node("a", seen=(0,)))] for _ in range(2)]
        self._step(b, is_first=[True, False])
        self.assertEqual(float(b._appearance[0]["a"][1].sum()), 0.0)
        self.assertEqual(float(b._appearance[1]["a"][1].sum()), DIM)

    def test_is_last_re_emits_and_leaves_the_cache_alone(self):
        b = _builder()
        b.plan = [[_graph(_node("a", seen=(0, 1)))]]
        first = self._step(b)
        cached = b._appearance[0]["a"].copy()
        b.dino.calls = 0
        again = self._step(b, is_last=[True])
        np.testing.assert_array_equal(
            again["graph_node_app"], first["graph_node_app"])
        np.testing.assert_array_equal(b._appearance[0]["a"], cached)
        self.assertEqual(b.dino.calls, 0)

    def test_a_mixed_batch_still_builds_the_live_environment(self):
        b = _builder(num_envs=2)
        b.plan = [[_graph(_node("a", seen=(0, 1)))],
                  [_graph(_node("b", seen=(0, 1)))]]
        self._step(b)
        b.plan = [[], [_graph(_node("b", seen=(1,)))]]
        b.dino.value = np.array([5.0, 5.0])
        out = self._step(b, is_last=[True, False])
        self.assertEqual(b.dino.calls, 2)
        self.assertEqual(len(b.plan[0]), 0)
        np.testing.assert_allclose(
            out["graph_node_app"][1, 0, 1], np.full(DIM, 5.0), atol=1e-2)

    def test_a_terminal_step_with_no_history_packs_zeros(self):
        b = _builder()
        out = b.step(is_last=[True])
        self.assertTrue((out["graph_node_ent"] == 0).all())


if __name__ == '__main__':
    unittest.main()
