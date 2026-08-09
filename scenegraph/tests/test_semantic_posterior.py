"""Invariance properties the semantic state depends on.

The graph token must not move when vertices are reordered or when the padding
width changes, and it must move when the valid content changes. Every mask the
posterior uses is derived from the packed content, so these also pin the
derivations themselves.
"""

import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import ninjax as nj
except ImportError:  # pragma: no cover - jax is optional for the sim tests
    jax = None

# Imported unguarded so a broken encoder raises here instead of hiding as a skip.
if jax is not None:
    from dreamerv3.graph_encoder import (
        GraphPosterior,
        _edge_selectors,
        _select_nodes,
        _sum_to_nodes,
        derive_masks,
        unpack,
    )


N_ENT = 8
CAMS = 2
APP_DIM = 6


def _graph(n_max, e_max, n_nodes, edges, order=None, seed=0, target=1):
    """Pack one batch element by hand. ``edges`` is a list of (src, dst, rel,
    abs, temp); ``order`` permutes the valid vertex slots; ``target`` names the
    goal vertex by its pre-permutation index, or None for no target."""
    rng = np.random.RandomState(seed)
    perm = list(order) if order is not None else list(range(n_nodes))
    slot = {old: new for new, old in enumerate(perm)}

    ent = np.zeros(n_max, np.uint16)
    node_app = np.zeros((n_max, CAMS, APP_DIM), np.float16)
    node_bbox = np.zeros((n_max, CAMS, 4), np.float16)
    node_target = np.zeros(n_max, np.uint8)
    # The default target applies only when that vertex exists; an empty graph
    # flags nothing, which is what the runtime packs too.
    if target is not None and target in slot:
        node_target[slot[target]] = 1
    base_app = rng.rand(n_nodes, CAMS, APP_DIM).astype(np.float16)
    xy = rng.rand(n_nodes, CAMS, 2).astype(np.float16) * 0.4
    for old in range(n_nodes):
        new = slot[old]
        ent[new] = 1 + old
        node_app[new] = base_app[old]
        for c in range(CAMS):
            # Every other vertex is invisible in the hand camera, which packs a
            # zero box; the head camera always sees it.
            if c and old % 2:
                continue
            x, y = xy[old, c]
            node_bbox[new, c] = [x, x + 0.3, y, y + 0.3]

    arrays = {
        'graph_edge_src': np.zeros(e_max, np.uint8),
        'graph_edge_dst': np.zeros(e_max, np.uint8),
        'graph_edge_rel': np.zeros(e_max, np.uint8),
        'graph_edge_abs': np.zeros(e_max, np.uint8),
        'graph_edge_temp': np.zeros(e_max, np.uint8),
    }
    for i, (s, d, rel, sig, tau) in enumerate(edges):
        arrays['graph_edge_src'][i] = slot[s]
        arrays['graph_edge_dst'][i] = slot[d]
        arrays['graph_edge_rel'][i] = rel
        arrays['graph_edge_abs'][i] = sig
        arrays['graph_edge_temp'][i] = tau

    out = {
        'graph_node_ent': ent, 'graph_node_app': node_app,
        'graph_node_bbox': node_bbox, 'graph_node_target': node_target,
        **arrays,
    }
    return {k: v[None] for k, v in out.items()}


EDGES = [
    (0, 1, 1, 1, 0),
    (0, 2, 5, 3, 2),
    (1, 2, 3, 2, 0),
]


@unittest.skipIf(jax is None, 'jax is not installed')
class DerivedMaskTests(unittest.TestCase):

    def _masks(self, graph):
        return jax.tree.map(np.asarray, derive_masks(unpack(graph)))

    def test_validity_follows_the_entity_id(self):
        m = self._masks(_graph(6, 8, 3, EDGES))
        np.testing.assert_array_equal(m['valid'][0], [1, 1, 1, 0, 0, 0])

    def test_camera_visibility_follows_the_box_extent(self):
        m = self._masks(_graph(6, 8, 3, EDGES))
        np.testing.assert_array_equal(m['camera_visible'][0, :3, 0], [1, 1, 1])
        np.testing.assert_array_equal(m['camera_visible'][0, :3, 1], [1, 0, 1])

    def test_a_one_pixel_box_near_the_frame_edge_reads_as_visible(self):
        # 1/112 of the frame at x ~ 0.9 is the tightest case the packed float16
        # box has to survive; deriving after a bf16 cast would be marginal.
        graph = _graph(6, 8, 3, EDGES)
        graph['graph_node_bbox'][0, 0, 0] = np.array(
            [100 / 112, 101 / 112, 100 / 112, 101 / 112], np.float16)
        m = self._masks(graph)
        self.assertEqual(float(m['camera_visible'][0, 0, 0]), 1.0)

    def test_appearance_support_follows_the_embedding_norm(self):
        graph = _graph(6, 8, 3, EDGES)
        graph['graph_node_app'][0, 1, 1] = 0.0
        m = self._masks(graph)
        self.assertEqual(float(m['appearance_known'][0, 1, 1]), 0.0)
        self.assertEqual(float(m['appearance_known'][0, 1, 0]), 1.0)

    def test_edge_masks_follow_the_relation_and_temporal_ids(self):
        m = self._masks(_graph(6, 8, 3, EDGES))
        np.testing.assert_array_equal(m['edge_valid'][0, :4], [1, 1, 1, 0])
        np.testing.assert_array_equal(m['temp_mask'][0, :4], [0, 1, 0, 0])


@unittest.skipIf(jax is None, 'jax is not installed')
class DenseEdgeContractionTests(unittest.TestCase):

    def test_matches_explicit_gather_and_scatter_with_duplicates_and_padding(self):
        nodes = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        src = np.array([[0, 1, 1, 0, 0], [3, 2, 0, 0, 0]], np.int32)
        dst = np.array([[1, 2, 2, 0, 0], [0, 1, 3, 0, 0]], np.int32)
        mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], np.float32)
        messages = (np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
                    - 5.0) * mask[..., None]

        source, destination = _edge_selectors(
            jnp.asarray(src), jnp.asarray(dst), jnp.asarray(mask),
            nodes.shape[1], jnp.float32)
        gathered = np.asarray(_select_nodes(source, jnp.asarray(nodes)))
        total = np.asarray(_sum_to_nodes(
            destination, jnp.asarray(messages)))
        count = np.asarray(destination.astype(jnp.float32).sum(1))

        expected_gathered = np.zeros_like(messages)
        expected_total = np.zeros_like(nodes)
        expected_count = np.zeros(nodes.shape[:2], np.float32)
        for batch in range(nodes.shape[0]):
            for edge in range(src.shape[1]):
                if not mask[batch, edge]:
                    continue
                expected_gathered[batch, edge] = nodes[batch, src[batch, edge]]
                expected_total[batch, dst[batch, edge]] += messages[batch, edge]
                expected_count[batch, dst[batch, edge]] += 1

        np.testing.assert_allclose(gathered, expected_gathered)
        np.testing.assert_allclose(total, expected_total)
        np.testing.assert_array_equal(count, expected_count)

    def test_count_accumulates_in_float32_past_bfloat16_integer_precision(self):
        edges = 540
        src = jnp.zeros((1, edges), jnp.int32)
        dst = jnp.full((1, edges), 2, jnp.int32)
        mask = jnp.ones((1, edges), jnp.float32)
        _, destination = _edge_selectors(
            src, dst, mask, num_nodes=4, dtype=jnp.bfloat16)
        count = destination.astype(jnp.float32).sum(1)

        self.assertEqual(count.dtype, jnp.float32)
        self.assertEqual(float(count[0, 2]), 540.0)


@unittest.skipIf(jax is None, 'jax is not installed')
class PoolingInvarianceTests(unittest.TestCase):

    def setUp(self):
        # app is the projection width and no longer has to match APP_DIM.
        self.model = GraphPosterior(
            layers=2, units=32, embed=8, app=8, bbox=4, entity_vocab=N_ENT,
            name='enc')
        # nj.pure takes the state positionally and the rng by keyword.
        self.fn = nj.pure(lambda g: self.model(g))
        base = _graph(6, 8, 3, EDGES)
        self.params, _ = self.fn(
            {}, unpack(base),
            seed=jax.random.PRNGKey(0), create=True, modify=True)

    def _token(self, graph):
        _, (_, token) = self.fn(
            self.params, unpack(graph), seed=jax.random.PRNGKey(0))
        return np.asarray(token, np.float32)

    def test_permuting_vertices_leaves_the_token_fixed(self):
        a = self._token(_graph(6, 8, 3, EDGES, order=[0, 1, 2]))
        b = self._token(_graph(6, 8, 3, EDGES, order=[2, 0, 1]))
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)

    def test_widening_the_padding_leaves_the_token_fixed(self):
        a = self._token(_graph(6, 8, 3, EDGES))
        b = self._token(_graph(12, 32, 3, EDGES))
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)

    def test_adding_a_real_vertex_moves_the_token(self):
        a = self._token(_graph(6, 8, 3, EDGES))
        b = self._token(_graph(6, 8, 4, EDGES + [(0, 3, 1, 2, 0)]))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def _perturbed(self, key, index, value):
        a = _graph(6, 8, 3, EDGES)
        b = {k: v.copy() for k, v in a.items()}
        b[key][index] = value
        return self._token(a), self._token(b)

    def test_head_appearance_moves_the_token(self):
        a, b = self._perturbed(
            'graph_node_app', (0, 1, 0), np.full(APP_DIM, 0.5, np.float16))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_hand_appearance_moves_the_token_independently(self):
        a, b = self._perturbed(
            'graph_node_app', (0, 1, 1), np.full(APP_DIM, 0.5, np.float16))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_moving_the_target_flag_moves_the_token(self):
        # Same vertices, same facts, different goal: the token has to separate
        # these or pick_all cannot tell one episode from another.
        a = self._token(_graph(6, 8, 3, EDGES, target=1))
        b = self._token(_graph(6, 8, 3, EDGES, target=2))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_dropping_the_target_flag_moves_the_token(self):
        a = self._token(_graph(6, 8, 3, EDGES, target=1))
        b = self._token(_graph(6, 8, 3, EDGES, target=None))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_bbox_moves_the_token(self):
        a, b = self._perturbed(
            'graph_node_bbox', (0, 1, 0),
            np.array([0.0, 1.0, 0.0, 1.0], np.float16))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_entity_id_moves_the_token(self):
        a, b = self._perturbed('graph_node_ent', (0, 2), np.uint16(N_ENT - 1))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)

    def test_padding_appearance_never_reaches_the_token(self):
        a, b = self._perturbed(
            'graph_node_app', (0, 4), np.full((CAMS, APP_DIM), 0.9, np.float16))
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)

    def test_making_an_unknown_camera_known_moves_the_token(self):
        """The gate rides on appearance_known, so a camera that starts with no
        embedding must start contributing the moment it has one."""
        a = _graph(6, 8, 3, EDGES)
        a['graph_node_app'][0, 1, 1] = 0.0
        b = {k: v.copy() for k, v in a.items()}
        b['graph_node_app'][0, 1, 1] = np.full(APP_DIM, 0.7, np.float16)
        self.assertGreater(
            float(np.abs(self._token(a) - self._token(b)).max()), 1e-3)

    def test_an_empty_graph_pools_to_exactly_zero(self):
        token = self._token(_graph(6, 8, 0, []))
        np.testing.assert_array_equal(token, np.zeros_like(token))


if __name__ == '__main__':
    unittest.main()
