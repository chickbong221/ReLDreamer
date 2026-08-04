"""Invariance properties the semantic state depends on.

The graph token must not move when vertices are reordered or when the padding
width changes, and it must move when the valid content changes -- pooling is
meant to preserve object-count information, not discard it.
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
    from dreamerv3.graph_encoder import GraphPosterior, unpack


N_ENT = 8
DETER = 16


def _graph(n_max, e_max, n_nodes, edges, feat=None, order=None, seed=0):
    """Pack one batch element by hand. ``edges`` is a list of (src, dst, rel,
    abs, temp, temp_mask); ``order`` permutes the valid vertex slots."""
    rng = np.random.RandomState(seed)
    perm = list(order) if order is not None else list(range(n_nodes))
    slot = {old: new for new, old in enumerate(perm)}

    ent = np.zeros(n_max, np.int32)
    vis = np.zeros(n_max, np.int32)
    valid = np.zeros(n_max, np.int32)
    node_feat = np.zeros((n_max, 2), np.float32)
    base_feat = feat if feat is not None else rng.rand(n_nodes, 2)
    for old in range(n_nodes):
        new = slot[old]
        ent[new] = 1 + old
        vis[new] = old % 2
        valid[new] = 1
        node_feat[new] = base_feat[old]

    arrays = {
        'graph_edge_src': np.zeros(e_max, np.int32),
        'graph_edge_dst': np.zeros(e_max, np.int32),
        'graph_edge_rel': np.zeros(e_max, np.int32),
        'graph_edge_abs': np.zeros(e_max, np.int32),
        'graph_edge_temp': np.zeros(e_max, np.int32),
        'graph_edge_temp_mask': np.zeros(e_max, np.int32),
        'graph_edge_valid': np.zeros(e_max, np.int32),
    }
    for i, (s, d, rel, sig, tau, mu) in enumerate(edges):
        arrays['graph_edge_src'][i] = slot[s]
        arrays['graph_edge_dst'][i] = slot[d]
        arrays['graph_edge_rel'][i] = rel
        arrays['graph_edge_abs'][i] = sig
        arrays['graph_edge_temp'][i] = tau
        arrays['graph_edge_temp_mask'][i] = mu
        arrays['graph_edge_valid'][i] = 1

    out = {
        'graph_node_ent': ent, 'graph_node_vis': vis,
        'graph_node_valid': valid, 'graph_node_feat': node_feat,
        'graph_n_nodes': np.int32(n_nodes),
        'graph_n_edges': np.int32(len(edges)),
        'graph_target_ent': np.int32(1),
        **arrays,
    }
    return {k: v[None] for k, v in out.items()}


EDGES = [
    (0, 1, 1, 1, 0, 0),
    (0, 2, 5, 3, 2, 1),
    (1, 2, 3, 2, 0, 0),
]


@unittest.skipIf(jax is None, 'jax is not installed')
class PoolingInvarianceTests(unittest.TestCase):

    def setUp(self):
        self.model = GraphPosterior(
            layers=2, units=32, embed=8, entity_vocab=N_ENT,
            condition_on_deter=True, name='enc')
        self.fn = nj.pure(lambda g, d: self.model(g, d))
        self.deter = jnp.zeros((1, DETER), jnp.float32)
        base = _graph(6, 8, 3, EDGES)
        self.params, _ = self.fn(
            {}, jax.random.PRNGKey(0), unpack(base), self.deter,
            create=True, modify=True)

    def _token(self, graph):
        _, (_, token) = self.fn(
            self.params, jax.random.PRNGKey(0), unpack(graph), self.deter)
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
        b = self._token(_graph(6, 8, 4, EDGES + [(0, 3, 1, 2, 0, 0)]))
        self.assertGreater(float(np.abs(a - b).max()), 1e-3)


if __name__ == '__main__':
    unittest.main()
