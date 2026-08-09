"""Semantic world model on synthetic data. No simulator, no assets.

Exercises everything the sim tests cannot reach: agent construction against the
real config at several model sizes, both the exact plain RSSM and its semantic
extension, the graph decoder heads and their masks, terminal masking, and the
fact that imagination runs without a graph at all.

Run on a machine with jax installed:

    python -m unittest scenegraph.tests.test_world_model
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
    import embodied.jax.nets as nn

    from dreamerv3.agent import Agent
    from dreamerv3.graph_encoder import GRAPH_KEYS, GraphDecoder, unpack
    from dreamerv3.rssm import Decoder, Encoder, RSSM

B, T, N, E = 2, 4, 5, 12
CAMS, APP_DIM = 2, 3
ENT_VOCAB = 6
IMAGE = (112, 112, 3)
PRESETS = ('size1m', 'size12m', 'size25m', 'size50m')


def _raw():
    root = pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3'
    return yaml.YAML(typ='safe').load((root / 'configs.yaml').read_text())


def _config(*presets, **overrides):
    """The agent config exactly as main.make_agent assembles it, including the
    whitelist-derived vocabulary size injected at startup."""
    raw = _raw()
    config = elements.Config(raw['defaults'])
    for name in presets:
        config = config.update(raw[name])
    config = config.update({'agent.graph.entity_vocab': ENT_VOCAB})
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


def _obs_space(app_dim=APP_DIM):
    return {
        'is_first': elements.Space(bool, ()),
        'is_last': elements.Space(bool, ()),
        'is_terminal': elements.Space(bool, ()),
        'reward': elements.Space(np.float32, ()),
        'image_head': elements.Space(np.uint8, IMAGE),
        'image_hand': elements.Space(np.uint8, IMAGE),
        'state': elements.Space(np.float32, (7,)),
        'graph_node_ent': elements.Space(np.uint16, (N,)),
        'graph_node_app': elements.Space(np.float16, (N, CAMS, app_dim)),
        'graph_node_bbox': elements.Space(np.float16, (N, CAMS, 4)),
        'graph_node_target': elements.Space(np.uint8, (N,)),
        'graph_edge_src': elements.Space(np.uint8, (E,)),
        'graph_edge_dst': elements.Space(np.uint8, (E,)),
        'graph_edge_rel': elements.Space(np.uint8, (E,)),
        'graph_edge_abs': elements.Space(np.uint8, (E,)),
        'graph_edge_temp': elements.Space(np.uint8, (E,)),
    }


def _graph_batch(rng, n_valid=N - 1, n_facts=E - 3):
    """(B, T, ...) packed arrays with a padded tail on both axes."""
    def tile(x):
        return np.broadcast_to(x, (B, T, *x.shape)).copy()

    ent = np.zeros(N, np.uint16)
    ent[:n_valid] = 1 + np.arange(n_valid) % (ENT_VOCAB - 1)

    app = np.zeros((N, CAMS, APP_DIM), np.float16)
    app[:n_valid] = rng.rand(n_valid, CAMS, APP_DIM).astype(np.float16)
    # Vertex 1 was never seen by the hand camera.
    app[1, 1] = 0.0

    bbox = np.zeros((N, CAMS, 4), np.float16)
    xy = rng.rand(n_valid, CAMS, 2).astype(np.float16) * 0.4
    for i in range(n_valid):
        for c in range(CAMS):
            # Vertex 2 is currently out of the hand view but keeps its cached
            # embedding, which is exactly the case the appearance loss covers.
            if c and i == 2:
                continue
            x, y = xy[i, c]
            bbox[i, c] = [x, x + 0.3, y, y + 0.3]

    # Vertex 3 is the subtask target; the ee at 0 never is.
    target = np.zeros(N, np.uint8)
    target[3] = 1

    src = np.zeros(E, np.uint8)
    dst = np.zeros(E, np.uint8)
    rel = np.zeros(E, np.uint8)
    sig = np.zeros(E, np.uint8)
    tau = np.zeros(E, np.uint8)
    src[:n_facts] = rng.randint(0, n_valid, n_facts)
    dst[:n_facts] = rng.randint(0, n_valid, n_facts)
    rel[:n_facts] = rng.randint(1, 11, n_facts)
    sig[:n_facts] = rng.randint(1, 3, n_facts)
    rel[0], rel[1] = 5, 1                # one delta-carrying fact, one without
    carries = rel[:n_facts] > 4          # spatial and affordance carry delta
    tau[:n_facts] = np.where(carries, rng.randint(1, 6, n_facts), 0)

    return {
        'graph_node_ent': tile(ent), 'graph_node_app': tile(app),
        'graph_node_bbox': tile(bbox), 'graph_node_target': tile(target),
        'graph_edge_src': tile(src), 'graph_edge_dst': tile(dst),
        'graph_edge_rel': tile(rel), 'graph_edge_abs': tile(sig),
        'graph_edge_temp': tile(tau),
    }


@unittest.skipIf(jax is None, 'jax is not installed')
class AgentConstructionTests(unittest.TestCase):
    """Catches config wiring without touching a device."""

    def setUp(self):
        self.config = _config('debug')
        self.act_space = {'action': elements.Space(np.float32, (4,), -1, 1)}

    def _agent(self, obs_space=None, config=None):
        return _model(
            obs_space if obs_space is not None else _obs_space(),
            self.act_space, config or self.config)

    def test_graph_keys_bypass_the_observation_encoder(self):
        agent = self._agent()
        self.assertTrue(agent.semantic)
        for key in agent.enc.veckeys + agent.enc.imgkeys:
            self.assertFalse(key.startswith('graph_'), key)
        for key in agent.dec.veckeys + agent.dec.imgkeys:
            self.assertFalse(key.startswith('graph_'), key)

    def test_the_instruction_conditions_the_encoder_but_is_not_reconstructed(self):
        space = _obs_space()
        space['instruction'] = elements.Space(np.float32, (8,))
        agent = self._agent(space)
        self.assertIn('instruction', agent.enc.veckeys)
        self.assertNotIn('instruction', agent.dec.veckeys)

    def test_both_cameras_reach_the_encoder_and_decoder(self):
        agent = self._agent()
        self.assertEqual(sorted(agent.enc.imgkeys), ['image_hand', 'image_head'])
        self.assertEqual(sorted(agent.dec.imgkeys), ['image_hand', 'image_head'])

    def test_scales_cover_every_graph_loss(self):
        agent = self._agent()
        for key in ('node', 'relabs', 'reltemp', 'semdyn', 'semrep'):
            self.assertIn(key, agent.scales)

    def test_suite_without_a_graph_drops_the_semantic_path(self):
        agent = self._agent({
            k: v for k, v in _obs_space().items()
            if not k.startswith('graph_')})
        self.assertFalse(agent.semantic)
        self.assertIsNone(agent.graphdec)
        self.assertNotIn('relabs', agent.scales)
        self.assertNotIn('sem', agent.dyn.entry_space)
        self.assertFalse(agent.dec.semantic)
        feat = dict(
            deter=jnp.zeros((2, agent.dyn.deter), jnp.float32),
            stoch=jnp.zeros(
                (2, agent.dyn.stoch, agent.dyn.classes), jnp.float32))
        self.assertEqual(
            agent.feat2tensor(feat).shape[-1],
            agent.dyn.deter + agent.dyn.stoch * agent.dyn.classes)

    def test_partial_graph_observation_space_is_rejected(self):
        space = _obs_space()
        del space[GRAPH_KEYS[-1]]
        with self.assertRaisesRegex(ValueError, 'Incomplete scene graph'):
            self._agent(space)

    def test_semantic_state_reaches_the_entry_space(self):
        self.assertIn('sem', self._agent().dyn.entry_space)

    def test_entity_vocab_reaches_the_posterior(self):
        self.assertEqual(
            self._agent().dyn.graph_kw['entity_vocab'], ENT_VOCAB)

    def test_app_dim_is_the_stored_width_not_the_projection(self):
        agent = self._agent()
        self.assertEqual(agent.graphdec.app_dim, self.config.graph.app_dim)
        self.assertNotEqual(
            self.config.graph.app_dim, self.config.graph.app)

    def test_the_posterior_is_not_handed_the_decoder_only_width(self):
        # Whatever graph_kw carries beyond GraphPosterior's own fields is
        # forwarded to its nn.Linear sublayers, which reject unknown kwargs.
        self.assertNotIn('app_dim', self._agent().dyn.graph_kw)


@unittest.skipIf(jax is None, 'jax is not installed')
class SizePresetTests(unittest.TestCase):
    """Nothing may hard-code a width: the size presets move all of them."""

    def test_every_preset_resolves_and_builds(self):
        act_space = {'action': elements.Space(np.float32, (4,), -1, 1)}
        widths = set()
        for name in PRESETS:
            config = _config(name)
            agent = _model(_obs_space(config.graph.app_dim), act_space, config)
            widths.add((config.graph.units, agent.enc.depths[-1]))
            self.assertEqual(agent.graphdec.app_dim, config.graph.app_dim)
        self.assertEqual(len(widths), len(PRESETS))

    def test_semantic_state_stays_16x16_outside_debug(self):
        for name in PRESETS:
            rssm = _config(name).dyn.rssm
            self.assertEqual(rssm['semstoch'], 16, name)
            self.assertEqual(rssm['semclasses'], 16, name)

    def test_the_image_path_reaches_a_legal_spatial_resolution(self):
        for name in PRESETS:
            config = _config(name)
            factor = 2 ** len(config.enc.simple.mults)
            minres = IMAGE[0] // factor
            self.assertEqual(IMAGE[0] % factor, 0, name)
            self.assertTrue(3 <= minres <= 16, (name, minres))


@unittest.skipIf(jax is None, 'jax is not installed')
class ImageShapeTests(unittest.TestCase):
    """112 has to survive both directions at whatever depth the preset picks."""

    def _spaces(self):
        return {k: elements.Space(np.uint8, IMAGE)
                for k in ('image_head', 'image_hand')}

    def _run(self, fn, *args):
        pure = nj.pure(fn)
        params, _ = pure({}, *args, seed=jax.random.PRNGKey(0),
                         create=True, modify=True)
        return pure(params, *args, seed=jax.random.PRNGKey(1))[1]

    def test_encoder_and_decoder_round_trip_both_cameras(self):
        rng = np.random.RandomState(0)
        enc = Encoder(self._spaces(), depth=4, mults=(2, 3, 4, 4), kernel=3,
                      layers=1, units=8, name='enc')
        dec = Decoder(self._spaces(), depth=4, mults=(2, 3, 4, 4), kernel=3,
                      layers=1, units=8, bspace=8, semantic=True, name='dec')
        obs = {k: jnp.asarray(rng.randint(0, 256, (B, T, *IMAGE), np.uint8))
               for k in self._spaces()}
        reset = jnp.zeros((B, T), bool)
        feat = dict(
            deter=jnp.zeros((B, T, 32), jnp.float32),
            stoch=jnp.zeros((B, T, 4, 4), jnp.float32),
            sem=jnp.zeros((B, T, 4, 4), jnp.float32))

        _, _, tokens = self._run(lambda: enc({}, obs, reset, True))
        # 112 -> 7 over four pooling stages, times whatever depth was resolved.
        self.assertEqual(tokens.shape[:2], (B, T))
        self.assertEqual(tokens.shape[-1], 7 * 7 * enc.depths[-1])

        _, _, recons = self._run(lambda: dec({}, feat, reset, True))
        for key in self._spaces():
            self.assertEqual(recons[key].pred().shape, (B, T, *IMAGE))

    def test_plain_decoder_needs_no_semantic_feature(self):
        dec = Decoder(self._spaces(), depth=4, mults=(2, 3, 4, 4), kernel=3,
                      layers=1, units=8, bspace=8, semantic=False, name='dec')
        reset = jnp.zeros((B, T), bool)
        feat = dict(
            deter=jnp.zeros((B, T, 32), jnp.float32),
            stoch=jnp.zeros((B, T, 4, 4), jnp.float32))
        _, _, recons = self._run(lambda: dec({}, feat, reset, True))
        for key in self._spaces():
            self.assertEqual(recons[key].pred().shape, (B, T, *IMAGE))


@unittest.skipIf(jax is None, 'jax is not installed')
class PlainWorldModelTests(unittest.TestCase):
    """The graph-off branch must retain the pre-semantic architecture."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.act_space = {
            'action': elements.Space(np.float32, (4,), -1, 1)}
        self.dyn = RSSM(
            self.act_space, semantic=False,
            deter=32, hidden=16, stoch=4, classes=4, blocks=4, name='dyn')
        self.tokens = jnp.asarray(rng.rand(B, T, 8), jnp.float32)
        self.acts = {'action': jnp.asarray(
            rng.rand(B, T, 4).astype(np.float32))}
        self.reset = jnp.zeros((B, T), bool).at[:, 0].set(True)

    def _create(self, fn):
        pure = nj.pure(fn)
        params, out = pure(
            {}, seed=jax.random.PRNGKey(0), create=True, modify=True)
        return params, out

    def test_state_losses_and_parameters_match_the_plain_rssm(self):
        self.assertEqual(set(self.dyn.entry_space), {'deter', 'stoch'})
        self.assertEqual(set(self.dyn.initial(B)), {'deter', 'stoch'})

        def fn():
            return self.dyn.loss(
                self.dyn.initial(B), self.tokens, {}, self.acts,
                self.reset, True)

        params, (_, entries, losses, feat, nodes, metrics) = self._create(fn)
        self.assertEqual(set(entries), {'deter', 'stoch'})
        self.assertEqual(set(feat), {'deter', 'stoch', 'logit'})
        self.assertEqual(set(losses), {'dyn', 'rep'})
        self.assertEqual(set(metrics), {'dyn_ent', 'rep_ent'})
        self.assertIsNone(nodes)

        semantic_names = ('dynin3', 'graphenc', 'semobs', 'semprior')
        for key in params:
            self.assertFalse(any(name in key for name in semantic_names), key)
        # Original Dreamer input per recurrent block: h/g + three projections
        # (h, z, action). The semantic model has a fourth projection.
        self.assertEqual(params['dyn/dynhid0/kernel'].shape[1], 32 // 4 + 3 * 16)

    def test_plain_imagination_carries_no_semantic_state(self):
        def fn():
            policy = {'action': jnp.zeros((B, 3, 4), jnp.float32)}
            return self.dyn.imagine(self.dyn.initial(B), policy, 3, False)

        _, (carry, feat, _) = self._create(fn)
        self.assertEqual(set(carry), {'deter', 'stoch'})
        self.assertEqual(set(feat), {'deter', 'stoch', 'logit'})


@unittest.skipIf(jax is None, 'jax is not installed')
class WorldModelTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.act_space = {'action': elements.Space(np.float32, (4,), -1, 1)}
        self.dyn = RSSM(
            self.act_space, graph_kw=dict(
                layers=1, units=16, embed=8, app=6, bbox=4,
                entity_vocab=ENT_VOCAB),
            deter=32, hidden=16, stoch=4, classes=4,
            semstoch=4, semclasses=4, semlayers=1, blocks=4, name='dyn')
        self.dec = GraphDecoder(
            APP_DIM, units=16, embed=8, entity_vocab=ENT_VOCAB,
            name='graphdec')
        self.graph = unpack(jax.tree.map(jnp.asarray, _graph_batch(rng)))
        self.tokens = jnp.asarray(rng.rand(B, T, 8), jnp.float32)
        self.acts = {'action': jnp.asarray(
            rng.rand(B, T, 4).astype(np.float32))}
        self.reset = jnp.zeros((B, T), bool).at[:, 0].set(True)
        self.live = jnp.ones((B, T), jnp.float32)

    def _run(self, fn, *args):
        # nj.pure takes the state positionally and the rng by keyword.
        pure = nj.pure(fn)
        params, _ = pure({}, *args, seed=jax.random.PRNGKey(0),
                         create=True, modify=True)
        return pure(params, *args, seed=jax.random.PRNGKey(1))[1]

    def _loss(self, step_valid=None):
        return self.dyn.loss(
            self.dyn.initial(B), self.tokens, self.graph, self.acts,
            self.reset, True,
            step_valid=self.live if step_valid is None else step_valid)

    def test_loss_emits_both_kl_pairs_at_batch_time_shape(self):
        _, _, losses, feat, nodes, metrics = self._run(self._loss)
        self.assertEqual(set(losses), {'dyn', 'rep', 'semdyn', 'semrep'})
        for key, value in losses.items():
            self.assertEqual(value.shape, (B, T), key)
        self.assertEqual(feat['sem'].shape[:2], (B, T))
        self.assertEqual(nodes.shape[:3], (B, T, N))
        self.assertIn('semdyn_raw', metrics)
        self.assertIn('semrep_raw', metrics)

    def test_raw_semantic_kl_is_measured_before_the_free_nats_floor(self):
        _, _, losses, _, _, metrics = self._run(self._loss)
        # The optimised loss is clipped at free_nats; the diagnostic is not, so
        # it is the only thing that can report a KL below the floor.
        for key in ('semdyn', 'semrep'):
            self.assertGreaterEqual(float(losses[key].min()), 1.0, key)
            self.assertLessEqual(
                float(metrics[f'{key}_raw']), float(losses[key].mean()), key)

    def test_terminal_steps_drop_the_semantic_kl(self):
        valid = jnp.ones((B, T), jnp.float32).at[:, -1].set(0.0)
        _, _, losses, _, _, _ = self._run(lambda: self._loss(valid))
        for key in ('semdyn', 'semrep'):
            np.testing.assert_allclose(
                np.asarray(losses[key])[:, -1], 0.0, atol=1e-6)
            self.assertGreater(float(losses[key][:, 0].min()), 0.0)
        # The low-level KL is untouched: the terminal image and state are real.
        _, _, losses, _, _, _ = self._run(lambda: self._loss(valid))
        self.assertGreater(float(losses['dyn'][:, -1].min()), 0.0)

    def test_a_zero_prediction_keeps_the_cosine_gradient_finite(self):
        # A padding node's representation is exactly zero, so a head's
        # zero-initialised bias makes its prediction exactly zero. The forward
        # cosine is finite either way; only the gradient shows a norm that was
        # floored after its square root instead of inside it.
        zero = jnp.zeros((2, 3), jnp.float32)
        for target in (zero, jnp.ones((2, 3), jnp.float32)):
            grad = jax.grad(
                lambda p, t=target: self.dec._cosine(p, t).sum())(zero)
            self.assertTrue(bool(jnp.isfinite(grad).all()))

    def test_decoder_gradients_are_finite_when_every_node_is_padding(self):
        def fn(nodes, sem):
            losses, _ = self.dec(nodes, self.graph, sem, self.live)
            return sum(v.sum() for v in losses.values())
        pure = nj.pure(fn)
        # The compute dtype, not float32: the decoder's layers assert on it,
        # and _loss() hands the real path nodes that are already cast.
        nodes = jnp.zeros((B, T, N, 16), nn.COMPUTE_DTYPE)
        sem = jnp.zeros((B, T, 4, 4), nn.COMPUTE_DTYPE)
        params, _ = pure({}, nodes, sem, seed=jax.random.PRNGKey(0),
                         create=True, modify=True)
        grads = jax.grad(
            lambda p: pure(p, nodes, sem, seed=jax.random.PRNGKey(1))[1])(params)
        for key, value in grads.items():
            self.assertTrue(np.isfinite(np.asarray(value)).all(), key)

    def test_decoder_heads_reduce_to_batch_time(self):
        def fn():
            _, _, _, feat, nodes, _ = self._loss()
            return self.dec(nodes, self.graph, feat['sem'], self.live)
        losses, metrics = self._run(fn)
        self.assertEqual(set(losses), {'node', 'relabs', 'reltemp', 'semtgt'})
        for key, value in losses.items():
            self.assertEqual(value.shape, (B, T), key)
            self.assertTrue(np.isfinite(np.asarray(value)).all(), key)
        for key, value in metrics.items():
            self.assertTrue(np.isfinite(float(value)), key)
        for key in ('node_vis_acc', 'relabs_acc', 'reltemp_acc',
                    'node_bbox_iou', 'semtgt_acc', 'semtgt_frac'):
            self.assertTrue(0.0 <= float(metrics[key]) <= 1.0, key)
        # Every frame in the fixture flags vertex 3, so the head is never
        # training on an all-masked batch.
        self.assertEqual(float(metrics['semtgt_frac']), 1.0)
        self.assertTrue(-1.0 <= float(metrics['node_app_cos']) <= 1.0)
        # The appearance target is random per vertex here, so a flat spread
        # would mean the masking collapsed it rather than the encoder.
        self.assertGreater(float(metrics['node_app_var']), 0.0)

    def test_a_frame_with_no_target_contributes_no_semtgt_loss(self):
        # Unresolved active objects fail open with an all-zero flag, so the
        # head must train on nothing there rather than on the pad entity.
        blind = dict(self.graph)
        blind['graph_node_target'] = jnp.zeros_like(blind['graph_node_target'])

        def fn():
            _, _, _, feat, nodes, _ = self._loss()
            return self.dec(nodes, blind, feat['sem'], self.live)
        losses, metrics = self._run(fn)
        np.testing.assert_allclose(
            np.asarray(losses['semtgt']), 0.0, atol=1e-6)
        self.assertEqual(float(metrics['semtgt_frac']), 0.0)
        self.assertTrue(np.isfinite(float(metrics['semtgt_acc'])))

    def test_terminal_steps_drop_every_graph_loss(self):
        valid = jnp.ones((B, T), jnp.float32).at[:, -1].set(0.0)

        def fn():
            _, _, _, feat, nodes, _ = self._loss()
            return self.dec(nodes, self.graph, feat['sem'], valid)
        losses, _ = self._run(fn)
        for key, value in losses.items():
            np.testing.assert_allclose(
                np.asarray(value)[:, -1], 0.0, atol=1e-6, err_msg=key)
            self.assertGreater(float(value[:, 0].min()), 0.0, key)

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
