"""Slot world model on synthetic data. No simulator, no assets.

Agent construction against the real config at several model sizes, both the
plain RSSM and its slot extension, the relation heads, and the fact that
imagination runs with no graph at all.

    python -m unittest scenegraph.tests.test_slot_model
"""

import pathlib
import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import elements
    import ruamel.yaml as yaml
except ImportError:  # pragma: no cover - jax is optional for the sim tests
    jax = None

if jax is not None:
    from dreamerv3.agent import Agent
    from dreamerv3.graph_encoder import GRAPH_KEYS

B, T, N, E = 2, 4, 6, 12
ENT_VOCAB, UID_VOCAB = 6, 16
IMAGE = (112, 112, 3)
PRESETS = ('size1m', 'size12m', 'size50m')
SLOT_LOSSES = ('slot', 'postabs', 'priorabs', 'posttemp', 'priortemp')


def _raw():
    root = pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3'
    return yaml.YAML(typ='safe').load((root / 'configs.yaml').read_text())


def _config(*presets, **overrides):
    raw = _raw()
    config = elements.Config(raw['defaults'])
    for name in presets:
        config = config.update(raw[name])
    config = config.update({
        'agent.graph.entity_vocab': ENT_VOCAB,
        'agent.graph.uid_vocab': UID_VOCAB,
    })
    if overrides:
        config = config.update(overrides)
    return elements.Config(
        **config.agent, logdir='', seed=0, jax=config.jax,
        batch_size=config.batch_size, batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length, replica=0, replicas=1)


def _model(obs_space, act_space, config):
    """Build the world model without the jax device wrapper.

    ``embodied.jax.Agent.__new__`` sets up meshes and devices before delegating
    to the model's own ``__init__``; this reproduces just that delegation, which
    is where all the config wiring lives.
    """
    model = object.__new__(Agent)
    model.__init__(obs_space, act_space, config)
    return model


def _obs_space(graph=True):
    space = {
        'is_first': elements.Space(bool, ()),
        'is_last': elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
        'reward': elements.Space(np.float32, ()),
        'image_head': elements.Space(np.uint8, IMAGE),
        'state': elements.Space(np.float32, (7,)),
    }
    if graph:
        space.update({
            'graph_node_ent': elements.Space(np.uint16, (N,)),
            'graph_node_uid': elements.Space(np.uint16, (N,)),
            'graph_node_target': elements.Space(np.uint8, (N,)),
            'graph_edge_src': elements.Space(np.uint8, (E,)),
            'graph_edge_dst': elements.Space(np.uint8, (E,)),
            'graph_edge_rel': elements.Space(np.uint8, (E,)),
            'graph_edge_abs': elements.Space(np.uint8, (E,)),
            'graph_edge_temp': elements.Space(np.uint8, (E,)),
        })
    return space


def _act_space():
    return {'action': elements.Space(np.float32, (4,), -1.0, 1.0)}


def _obs(rng, space, valid=4, facts=8):
    """One batch with the padding laid out the way the packer writes it."""
    out = {}
    for key, sp in space.items():
        if key.startswith('graph_'):
            continue
        shape = (B, T, *sp.shape)
        if sp.dtype == np.uint8:
            out[key] = rng.integers(0, 255, shape, np.uint8)
        elif sp.dtype == bool:
            out[key] = np.zeros(shape, bool)
        else:
            out[key] = rng.random(shape).astype(np.float32)
    if 'graph_node_ent' not in space:
        return out
    ent = np.zeros((B, T, N), np.uint16)
    ent[:, :, :valid] = rng.integers(1, ENT_VOCAB, (B, T, valid))
    uid = np.zeros((B, T, N), np.uint16)
    # Slot 0 is the ee; object uids are stable across the chunk.
    uid[:, :, 0] = 1
    uid[:, :, 1:valid] = rng.integers(2, UID_VOCAB, (B, 1, valid - 1))
    target = np.zeros((B, T, N), np.uint8)
    target[:, :, 1] = 1
    column = lambda hi: np.concatenate([
        rng.integers(1, hi, (B, T, facts)).astype(np.uint8),
        np.zeros((B, T, E - facts), np.uint8)], -1)
    out.update({
        'graph_node_ent': ent,
        'graph_node_uid': uid,
        'graph_node_target': target,
        'graph_edge_src': rng.integers(0, valid, (B, T, E)).astype(np.uint8),
        'graph_edge_dst': rng.integers(0, valid, (B, T, E)).astype(np.uint8),
        'graph_edge_rel': column(3),
        'graph_edge_abs': column(3),
        'graph_edge_temp': column(3),
    })
    return out


def _run(model, obs, act_space):
    carry = model.init_train(B)[:3]
    prevact = {
        k: jnp.zeros((B, T, *v.shape), jnp.float32)
        for k, v in act_space.items()}
    fn = lambda: model.loss(carry, obs, prevact, training=True)
    _, out = nj.pure(fn)(
        {}, seed=jax.random.PRNGKey(0), create=True, modify=True)
    return out


@unittest.skipIf(jax is None, 'jax not installed')
class Construction(unittest.TestCase):

    def test_every_size_preset_builds_with_slots(self):
        for preset in PRESETS:
            with self.subTest(preset):
                model = _model(_obs_space(), _act_space(), _config(preset))
                self.assertTrue(model.semantic)

    def test_the_plain_model_builds_with_no_graph_keys(self):
        model = _model(_obs_space(graph=False), _act_space(), _config('size1m'))
        self.assertFalse(model.semantic)
        self.assertIsNone(model.graphdec)
        for key in SLOT_LOSSES:
            self.assertNotIn(key, model.scales)

    def test_a_partial_graph_space_fails_loud(self):
        space = _obs_space()
        space.pop('graph_node_uid')
        with self.assertRaisesRegex(ValueError, 'Incomplete scene graph'):
            _model(space, _act_space(), _config('size1m'))

    def test_slot_width_survives_the_size100m_wildcard(self):
        # size100m carries `.*\.units: 768`. The graph branch must not have a
        # key by that name, or it silently widens to the preset's model width.
        raw = _raw()
        config = elements.Config(raw['defaults']).update(raw['size100m'])
        self.assertNotIn('units', dict(config.agent.graph))
        self.assertEqual(config.agent.dyn.rssm.slot_dim, 256)
        self.assertEqual(config.agent.dyn.rssm.slots, 6)

    def test_the_packed_capacity_matches_the_slot_count(self):
        raw = _raw()
        config = elements.Config(raw['defaults'])
        self.assertEqual(
            config.env.maniskill.graph.n_max, config.agent.dyn.rssm.slots)


@unittest.skipIf(jax is None, 'jax not installed')
class Losses(unittest.TestCase):

    def setUp(self):
        self.space = _obs_space()
        self.act = _act_space()
        self.config = _config('size1m')
        self.model = _model(self.space, self.act, self.config)
        rng = np.random.default_rng(0)
        self.obs = jax.tree.map(jnp.asarray, _obs(rng, self.space))

    def test_every_declared_loss_is_produced_at_batch_shape(self):
        _, (_, _, outs, _) = _run(self.model, self.obs, self.act)
        losses = outs['losses']
        self.assertEqual(set(losses), set(self.model.scales))
        for key in SLOT_LOSSES:
            self.assertEqual(losses[key].shape, (B, T))
            self.assertTrue(np.isfinite(np.asarray(losses[key])).all())

    def test_a_terminal_frame_contributes_no_slot_loss(self):
        obs = dict(self.obs)
        obs['is_last'] = jnp.asarray(
            np.tile([False, False, False, True], (B, 1)))
        _, (_, _, outs, _) = _run(self.model, obs, self.act)
        for key in SLOT_LOSSES:
            self.assertTrue(
                np.allclose(np.asarray(outs['losses'][key])[:, -1], 0.0),
                key)

    def test_an_empty_graph_leaves_every_slot_zero(self):
        obs = dict(self.obs)
        for key in GRAPH_KEYS:
            obs[key] = jnp.zeros_like(obs[key])
        _, (_, _, outs, _) = _run(self.model, obs, self.act)
        slots = np.asarray(outs['repfeat']['slots'])
        self.assertTrue((slots == 0).all())
        self.assertFalse(np.asarray(outs['repfeat']['slot_mask']).any())


if __name__ == '__main__':
    unittest.main()
