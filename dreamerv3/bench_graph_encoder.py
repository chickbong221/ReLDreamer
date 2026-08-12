"""Wall-clock cost of the graph posterior at training shapes.

The posterior runs on every replay timestep, so one train step pays it
``batch_size * batch_length`` times; the training loop only reports the total.
This isolates it at the shapes ``maniskill_rgb mshab`` actually trains on --
batch 32 x length 64 flattened to B, n_max 10, e_max 270, and size100m's width
of 768 -- and prints forward and forward+backward milliseconds.

Nothing here touches replay or the simulator: the graph is synthetic, with a
fixed valid-vertex and valid-fact count, so two revisions see identical work.
Only ``GraphPosterior`` and ``unpack`` are read, and both predate the EGT
rewrite, so an older encoder can be swapped under this same script:

    python -m dreamerv3.bench_graph_encoder
    git checkout <pre-egt-commit> -- dreamerv3/graph_encoder.py
    python -m dreamerv3.bench_graph_encoder
    git checkout HEAD -- dreamerv3/graph_encoder.py
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from .graph_encoder import GraphPosterior, unpack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=32 * 64,
                   help='batch_size * batch_length, which is what the encoder '
                        'sees in one call')
    p.add_argument('--n-max', type=int, default=10)
    p.add_argument('--e-max', type=int, default=270)
    p.add_argument('--valid', type=int, default=6,
                   help='valid vertices; the rest of n_max is padding')
    p.add_argument('--edges', type=int, default=200,
                   help='valid facts; the rest of e_max is padding')
    p.add_argument('--units', type=int, default=768,
                   help='agent.graph.units; size100m resolves it to 768')
    p.add_argument('--app-dim', type=int, default=384)
    p.add_argument('--repeats', type=int, default=50)
    return p.parse_args()


def make_batch(rng, args, cams=2):
    """One packed batch with the padding laid out the way the runtime packs it.

    Valid vertices fill a prefix and padding keeps entity id zero; valid facts
    fill a prefix and padding keeps relation id zero. Every mask the posterior
    reads is derived from those two, so this exercises the real masked paths.
    """
    B, N, E, V = args.batch, args.n_max, args.e_max, args.valid

    ent = np.zeros((B, N), np.uint16)
    ent[:, :V] = rng.randint(1, 64, (B, V))
    app = np.zeros((B, N, cams, args.app_dim), np.float16)
    app[:, :V] = rng.rand(B, V, cams, args.app_dim).astype(np.float16)
    bbox = np.zeros((B, N, cams, 4), np.float16)
    xy = rng.rand(B, V, cams, 2).astype(np.float16) * 0.4
    bbox[:, :V, :, 0] = xy[..., 0]
    bbox[:, :V, :, 1] = xy[..., 0] + 0.3
    bbox[:, :V, :, 2] = xy[..., 1]
    bbox[:, :V, :, 3] = xy[..., 1] + 0.3
    target = np.zeros((B, N), np.uint8)
    target[:, 0] = 1

    def column(classes):
        out = np.zeros((B, E), np.uint8)
        out[:, :args.edges] = rng.randint(1, classes, (B, args.edges))
        return out

    return {
        'graph_node_ent': ent,
        'graph_node_app': app,
        'graph_node_bbox': bbox,
        'graph_node_target': target,
        'graph_edge_src': rng.randint(0, V, (B, E)).astype(np.uint8),
        'graph_edge_dst': rng.randint(0, V, (B, E)).astype(np.uint8),
        'graph_edge_rel': column(6),
        'graph_edge_abs': column(4),
        'graph_edge_temp': column(3),
    }


def time_call(fn, repeats):
    """Median of ``repeats`` timed calls, after five warmups for compilation."""
    for _ in range(5):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e3


def main() -> int:
    args = parse_args()
    if args.valid > args.n_max or args.edges > args.e_max:
        print('FAIL: valid vertices or facts exceed the padded capacity')
        return 1

    rng = np.random.RandomState(0)
    graph = unpack(jax.tree.map(
        jnp.asarray, make_batch(rng, args)))
    model = GraphPosterior(
        units=args.units, entity_vocab=64, winit='trunc_normal_in', name='enc')

    def forward(g):
        return model(g)

    def lossfn(g):
        nodes, token = model(g)
        return (nodes.astype(jnp.float32).sum() +
                token.astype(jnp.float32).sum())

    def backward(g):
        # The posterior sits under the world-model optimizer, so the number
        # that matters is the one that includes its backward pass.
        _, _, grads = nj.grad(lossfn, [model])(g)
        return grads

    params, _ = nj.pure(forward)(
        {}, graph, seed=jax.random.PRNGKey(0), create=True, modify=True)
    count = sum(int(np.prod(v.shape)) for v in jax.tree.leaves(params))

    seed = jax.random.PRNGKey(0)
    fwd = jax.jit(lambda p, g: nj.pure(forward)(p, g, seed=seed)[1])
    bwd = jax.jit(lambda p, g: nj.pure(backward)(p, g, seed=seed)[1])
    forward_ms = time_call(lambda: fwd(params, graph), args.repeats)
    both_ms = time_call(lambda: bwd(params, graph), args.repeats)

    print(f'B={args.batch} n_max={args.n_max} e_max={args.e_max} '
          f'valid={args.valid} facts={args.edges} units={args.units}')
    print(f'params            {count / 1e6:8.3f} M')
    print(f'forward           {forward_ms:8.2f} ms')
    print(f'forward+backward  {both_ms:8.2f} ms')
    return 0


if __name__ == '__main__':
    sys.exit(main())
