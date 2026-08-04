"""Semantic world model on synthetic data. No simulator, no assets.

Exercises everything the sim tests cannot reach: agent construction against the
real config, the RSSM with a semantic state, the graph decoder heads, and the
fact that imagination runs without a graph at all.

Run on a machine with jax installed:

    python -m unittest teemo_sim_probe.tests.test_world_model
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
    from dreamerv3.graph_encoder import GraphDecoder, unpack
    from dreamerv3.rssm import RSSM

B, T, N, E, F = 2, 4, 5, 12, 2
ENT_VOCAB = 6
DEPTH = (16, 16, 1)


def _config():
    """The agent config exactly as main.make_agent assembles it, including the
    whitelist-derived vocabulary size injected at startup."""
    root = pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3'
    raw = yaml.YAML(typ='safe').load((root / 'configs.yaml').read_text())
    config = elements.Config(raw['defaults']).update(raw['debug'])
    config = config.update({'agent.graph.entity_vocab': ENT_VOCAB})
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


def _obs_space():
    space = {
        'is_first': elements.Space(bool, ()),
        'is_last': elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
        'reward': elements.Space(np.float32, ()),
        'depth_head': elements.Space(np.uint16, DEPTH, 0, 20000),
        'depth_hand': elements.Space(np.uint16, DEPTH, 0, 20000),
        'state': elements.Space(np.float32, (7,)),
        'graph_node_ent': elements.Space(np.uint16, (N,)),
        'graph_node_vis': elements.Space(np.uint8, (N,)),
        'graph_node_valid': elements.Space(np.uint8, (N,)),
        'graph_node_feat': elements.Space(np.float32, (N, F)),
        'graph_edge_src': elements.Space(np.uint8, (E,)),
        'graph_edge_dst': elements.Space(np.uint8, (E,)),
        'graph_edge_rel': elements.Space(np.uint8, (E,)),
        'graph_edge_abs': elements.Space(np.uint8, (E,)),
        'graph_edge_temp': elements.Space(np.uint8, (E,)),
        'graph_edge_temp_mask': elements.Space(np.uint8, (E,)),
        'graph_edge_valid': elements.Space(np.uint8, (E,)),
        'graph_n_nodes': elements.Space(np.int32, ()),
        'graph_n_edges': elements.Space(np.int32, ()),
        'graph_target_ent': elements.Space(np.uint16, ()),
    }
    return space


def _graph_batch(rng, n_valid=N - 1, n_facts=E - 3):
    """(B, T, ...) packed arrays with a padded tail on both axes."""
    def tile(x):
        return np.broadcast_to(x, (B, T, *x.shape)).copy()

    ent = np.zeros(N, np.int32)
    vis = np.zeros(N, np.int32)
    valid = np.zeros(N, np.int32)
    ent[:n_valid] = 1 + np.arange(n_valid) % (ENT_VOCAB - 1)
    vis[:n_valid] = rng.randint(0, 2, n_valid)
    valid[:n_valid] = 1
    feat = np.zeros((N, F), np.float32)
    feat[:n_valid] = rng.rand(n_valid, F).astype(np.float32)

    src = np.zeros(E, np.int32)
    dst = np.zeros(E, np.int32)
    rel = np.zeros(E, np.int32)
    sig = np.zeros(E, np.int32)
    tau = np.zeros(E, np.int32)
    tmask = np.zeros(E, np.int32)
    emask = np.zeros(E, np.int32)
    src[:n_facts] = rng.randint(0, n_valid, n_facts)
    dst[:n_facts] = rng.randint(0, n_valid, n_facts)
    rel[:n_facts] = rng.randint(1, 11, n_facts)
    sig[:n_facts] = rng.randint(1, 3, n_facts)
    emask[:n_facts] = 1
    carries = rel[:n_facts] > 4          # spatial and affordance carry delta
    tau[:n_facts] = np.where(carries, rng.randint(1, 6, n_facts), 0)
    tmask[:n_facts] = carries.astype(np.int32)

    return {
        'graph_node_ent': tile(ent), 'graph_node_vis': tile(vis),
        'graph_node_valid': tile(valid), 'graph_node_feat': tile(feat),
        'graph_edge_src': tile(src), 'graph_edge_dst': tile(dst),
        'graph_edge_rel': tile(rel), 'graph_edge_abs': tile(sig),
        'graph_edge_temp': tile(tau), 'graph_edge_temp_mask': tile(tmask),
        'graph_edge_valid': tile(emask),
        'graph_n_nodes': np.full((B, T), n_valid, np.int32),
        'graph_n_edges': np.full((B, T), n_facts, np.int32),
        'graph_target_ent': np.full((B, T), 1, np.int32),
    }


@unittest.skipIf(jax is None, 'jax is not installed')
class AgentConstructionTests(unittest.TestCase):
    """Catches config wiring without touching a device."""

    def setUp(self):
        self.config = _config()
        self.act_space = {'action': elements.Space(np.float32, (4,), -1, 1)}

    def _agent(self, obs_space=None):
        return _model(
            obs_space if obs_space is not None else _obs_space(),
            self.act_space, self.config)

    def test_graph_keys_bypass_the_observation_encoder(self):
        agent = self._agent()
        self.assertTrue(agent.semantic)
        for key in agent.enc.veckeys + agent.enc.imgkeys:
            self.assertFalse(key.startswith('graph_'), key)
        for key in agent.dec.veckeys + agent.dec.imgkeys:
            self.assertFalse(key.startswith('graph_'), key)

    def test_scales_cover_every_semantic_loss(self):
        agent = self._agent()
        for key in ('semapp', 'semvis', 'semabs', 'semtemp',
                    'semdyn', 'semrep'):
            self.assertIn(key, agent.scales)

    def test_suite_without_a_graph_drops_the_semantic_path(self):
        agent = self._agent({
            k: v for k, v in _obs_space().items()
            if not k.startswith('graph_')})
        self.assertFalse(agent.semantic)
        self.assertIsNone(agent.graphdec)
        self.assertNotIn('semabs', agent.scales)

    def test_semantic_state_reaches_the_entry_space(self):
        self.assertIn('sem', self._agent().dyn.entry_space)

    def test_entity_vocab_reaches_the_posterior(self):
        self.assertEqual(
            self._agent().dyn.graph_kw['entity_vocab'], ENT_VOCAB)


@unittest.skipIf(jax is None, 'jax is not installed')
class WorldModelTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.act_space = {'action': elements.Space(np.float32, (4,), -1, 1)}
        self.dyn = RSSM(
            self.act_space, graph_kw=dict(
                layers=1, units=16, embed=8, entity_vocab=ENT_VOCAB,
                condition_on_deter=True),
            deter=32, hidden=16, stoch=4, classes=4,
            semstoch=4, semclasses=4, semlayers=1, blocks=4, name='dyn')
        self.dec = GraphDecoder(F, units=16, embed=8, name='graphdec')
        self.graph = unpack(jax.tree.map(jnp.asarray, _graph_batch(rng)))
        self.tokens = jnp.asarray(rng.rand(B, T, 8), jnp.float32)
        self.acts = {'action': jnp.asarray(
            rng.rand(B, T, 4).astype(np.float32))}
        self.reset = jnp.zeros((B, T), bool).at[:, 0].set(True)

    def _run(self, fn, *args):
        # nj.pure takes the state positionally and the rng by keyword.
        pure = nj.pure(fn)
        params, _ = pure({}, *args, seed=jax.random.PRNGKey(0),
                         create=True, modify=True)
        return pure(params, *args, seed=jax.random.PRNGKey(1))[1]

    def test_loss_emits_both_kl_pairs_at_batch_time_shape(self):
        _, _, losses, feat, nodes, _ = self._run(
            lambda: self.dyn.loss(
                self.dyn.initial(B), self.tokens, self.graph, self.acts,
                self.reset, True))
        self.assertEqual(set(losses), {'dyn', 'rep', 'semdyn', 'semrep'})
        for key, value in losses.items():
            self.assertEqual(value.shape, (B, T), key)
        self.assertEqual(feat['sem'].shape[:2], (B, T))
        self.assertEqual(nodes.shape[:3], (B, T, N))

    def test_decoder_heads_reduce_to_batch_time(self):
        def fn():
            _, _, _, _, nodes, _ = self.dyn.loss(
                self.dyn.initial(B), self.tokens, self.graph, self.acts,
                self.reset, True)
            return self.dec(nodes, self.graph)
        losses, metrics = self._run(fn)
        self.assertEqual(
            set(losses), {'semapp', 'semvis', 'semabs', 'semtemp'})
        for key, value in losses.items():
            self.assertEqual(value.shape, (B, T), key)
            self.assertTrue(np.isfinite(np.asarray(value)).all(), key)
        for key, value in metrics.items():
            self.assertTrue(0.0 <= float(value) <= 1.0, key)

    def test_imagination_needs_no_graph(self):
        def fn():
            carry = self.dyn.initial(B)
            policy = {'action': jnp.zeros((B, 3, 4), jnp.float32)}
            return self.dyn.imagine(carry, policy, 3, False)
        _, feat, _ = self._run(fn)
        self.assertEqual(feat['sem'].shape[:2], (B, 3))
        self.assertTrue(np.isfinite(np.asarray(feat['deter'])).all())


if __name__ == '__main__':
    unittest.main()
