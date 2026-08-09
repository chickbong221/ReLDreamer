# Scene graph

Build a fixed-size manipulation graph from MS-HAB segmentation + physics, one
per simulator step, and pack it into the replay contract the world model reads.
MS-HAB is read-only; all integration lives under `scenegraph/adapters/`.

```text
  ─── offline, tools/ (prepare_assets runs all of it) ───────────────

   MS-HAB env  ──►  FetchCollectContactDataWrapper  ──►  <obj>.pkl
                                                          │
                          ┌───────────────────────────────┤
                          ▼                               ▼
                  build_affordances              build_subtask_whitelists
                          │                               │
                          ▼                               ▼
                  affordances.json              subtask_whitelists/*.json

   task plans  ──►  build_instruction_embeddings  ──►  instructions.npz

  ─── online, here (every simulator step) ───────────────────────────

   MS-HAB env  ──►  GraphBuilder ◄── affordances.json + whitelist
                          │
                          ▼
                  Graph(nodes, edges)
                          │
        frozen DINOv2 ──► │ per-camera pooling + appearance cache
                          ▼
                  packed replay arrays
```

Each step the runtime gates nodes via the whitelist, scores compatibilities
against the affordance asset, emits absolute and temporal edges, pools one
frozen DINOv2 embedding per node per camera, and packs the result.

## Related packages

* **`teemo_sim_probe/`** is a frozen, self-contained demo of the predecessor
  pipeline — single camera, mean masked depth per node, target-conditioned
  pooling — kept runnable for figures. It carries its own copy of every module
  it needs, including the mining tools, so it never has to track changes here.
  Nothing in this package imports it.
* **`dreamerv3/`** consumes the packed arrays: `graph_encoder.py` holds the
  semantic posterior and graph decoder. See its README for the model side,
  the loss scales, and what to watch during training.

## Replay contract

Two cameras, stored separately and never fused. Camera index 0 is
`fetch_head`, 1 is `fetch_hand`.

```text
graph_node_ent     [10]          uint16
graph_node_app     [10,2,384]    float16
graph_node_bbox    [10,2,4]      float16
graph_node_target  [10]          uint8
graph_edge_src     [96]          uint8
graph_edge_dst     [96]          uint8
graph_edge_rel     [96]          uint8
graph_edge_abs     [96]          uint8
graph_edge_temp    [96]          uint8
```

Nothing derivable is stored — no masks, no counts. Index zero is padding in
every vocabulary, so the model reads validity, per-camera visibility,
per-camera appearance support and both counts back off the ids, the boxes and
the embedding norms. `dreamerv3/graph_encoder.derive_masks` is the single place
those derivations live.

`graph_node_target` is the exception that proves the rule: which vertex the
current subtask acts on is not derivable from anything else in the pack, and
under `mshab_obj: all` it changes from subtask to subtask.

Appearance comes from frozen `dinov2_vits14_reg` at 112px (8x8 patches),
pooled under each camera's fractional patch coverage, in the collector only.
Nothing the agent trains can move it, so a vector written to replay stays
valid for the whole run.

## Graph contract

Two node types:

* `ee` -- end effector (tcp + finger1 + finger2 folded into one node), always
  vertex index 0.
* `object` -- every non-robot actor or articulation link.

The vertex set is append-only per episode: an index is handed out on first
sight and never reused or reordered, so a node that leaves the view keeps its
position and its last pose. Each node carries an entity id plus, per camera, a
normalised bounding box and a frozen DINOv2 embedding.

Visibility is per camera and immediate: one segmentation pixel makes a node
visible in that camera, with no area threshold and no grace frames. A node
visible in either camera is globally visible; one visible in neither stays a
valid vertex for the rest of the episode. Boxes always describe the current
frame and go to zero for a camera that cannot see the node. Embeddings persist:
a camera that loses sight keeps its last one, and a camera that has never seen
the node holds exactly zero. Retention lives in the adapter-level cache, not in
the registry.

A node appears only if it is listed in the episode's per-`(subtask, target)`
whitelist, which holds the target plus the entities that directly support it.
Contact alone does not admit anything: a rollout touches whatever is in the
way, so admitting every contacted entity fills a pick-the-bowl graph with the
groceries the arm brushed past.

### Relation vocabulary

One fact per admissible `(src, relation, dst)`, carrying an absolute state
`sigma` and, where the family defines one, a change `delta` over a `K`-frame
window.

| Family | Pair type | Relation | `sigma` | `delta` |
|---|---|---|---|---|
| **Physical state** | ee--obj | `grasp` | `not-holds` / `holds` | -- |
|  | ee--obj, obj--obj | `contact` | (same) | -- |
|  | obj--obj (directed) | `support` | (same) | -- |
|  | obj--obj (directed) | `contain` | (same) | -- |
| **Spatial** | ee--obj | `planar-distance` | 5 distance bins | 5-way signed |
|  | ee--obj | `height-offset` | 5 height bins | 5-way signed |
| **Affordance** | ee--obj | `grasp-compatibility` | `match` / `partial-match` / `poor-match` / `unobserved` | 5-way signed |
|  | ee--obj, obj--obj | `contact-compatibility` | (same) | (same) |
|  | obj--obj (directed) | `support-compatibility` | (same) | (same) |
|  | obj--obj (directed) | `contain-compatibility` | (same) | (same) |

`delta` is a shared signed vocabulary (`decrease-fast` .. `increase-fast`);
index 0 is always the most negative change, so it reads as approaching for
distances and as fitting better for mismatch scores.

### Gating rules

1. **Both endpoints must be visible.** The only exception is object--object
   physical state, which is retained at its last observed value -- both
   polarities -- while an endpoint is out of view.
2. **Physical-state relations are independent.** A grasped object reports
   `grasp: holds` and `contact: holds` at once; a supported object in contact
   with its supporter reports both. There is no priority chain.
3. **Whitelist `interaction_types` gate admissibility.** ee--obj checks the
   object's tokens, obj--obj checks both endpoints'.
4. **Spatial is object-center**, ungated, for every visible `ee--obj` pair.
5. **The affordance near gate picks a label, not a fact.** The score is
   measured for every admissible instance; failing the gate emits
   `unobserved` while still accumulating the change, so an approach is visible
   before the absolute label flips. Object--object pairs beyond
   `object_object_compat_max_distance` skip scoring entirely.
6. **`delta` is masked** only for the physical-state family and for facts with
   fewer than `K + 1` samples.

A retained fact keeps its last observed absolute state and never gains a
temporal label, because temporal annotation skips stale edges and the cached
copy is taken before annotation runs. The world model trains the absolute head
on those edges and masks the temporal one.

### Compatibility scoring

Score = unweighted mean of `[0, 1]` per-component mismatches, binned at
`[1/3, 2/3]`. Per-relation components:

* `grasp-compatibility` (ee → near_obj): `pos`, `orient`, `width`.
* `contact-compatibility`:
  * ee--obj: `pos`, `orient` against the active grasp anchor.
  * obj--obj: `pos` between matched contact anchors; `orient` between each
    side's outward normal (anti-parallel at a real contact).
* `support-compatibility` (supporter → supported): `xy` (clipped inside
  `footprint_radius`), `vertical`, `orient`.
* `contain-compatibility` (container → containee, PegInsertion template):
  `radial` (past `opening_radius`), `axial` (past `[0, depth]`), `orient`.

Normalizers live under `cfg["compat_norm"]` (defaults in
`relation_rules._compat_norm`, overridable via `configs/thresholds.yaml`).

### Bin edges

`planar-distance`, `height-offset`, and every `*-change` relation use equal-
width splits of `[0, max]` (unsigned) or `[-max, max]` (signed), where `max`
is the 0.9 quantile across demo samples. The runtime reads one bin set per
subtask, from `<subtask>_all.json` — per-target edges would leave the same
relation token meaning a different metric distance in each episode.
Compatibility absolute edges are fixed at `[1/3, 2/3]` (score is already in
`[0, 1]`). `configs/thresholds.yaml` provides fallbacks for relations the
asset omits.

## Asset shapes

Whitelist (schema v4, `<subtask>_<target>.json`):

```text
{
  subtask, target,
  members:   {<key>: {roles, interaction_types, kind, name?, ...}},
  bin_edges: {<relation>: [edges...]}
}
```

Affordances (schema v3, `affordances.json`), keyed by canonical object id:

```text
{<key>: {
  grasp_components:   [{anchor, approach_dir, width}],
  contact_components: [{anchor, outward_normal}],
  support_components: [{surface_anchor, surface_normal, footprint_radius}],
  bottom_components:  [{bottom_anchor, bottom_normal}],
  contain_components: [{entry_anchor, entry_axis, opening_radius, depth}],
  key_components:     [{key_anchor, key_axis}]
}}
```

Both are produced by `tools/`, along with the rollout pickles they are mined
from, and land in `configs/` next to `thresholds.yaml`, which points at them by
relative path. `env.maniskill.graph.whitelist_dir` overrides the whitelist half.

The runtime fails loud at episode start when no matching whitelist exists for
`(subtask, target)`. Re-mine everything with:

```bash
python -m scenegraph.tools.prepare_assets \
  --mshab-task set_table prepare_groceries tidy_house --subtask pick --clean
```

## Tests

```bash
python -m unittest discover scenegraph/tests
```

Covers the graph contract, the frozen encoder's batching and pooling (with the
checkpoint mocked, so no weights are downloaded), the per-camera appearance
cache and the terminal guard, and — on a machine with jax — the semantic
posterior, the graph decoder's masks, and agent construction across model-size
presets. The torch and jax suites **skip silently** without those libraries, so
confirm the run reports no skips before trusting it.

The only thing that touches a simulator:

```bash
python -m scenegraph.tools.smoke_graph_obs --num-envs 4 --steps 120
```

Use at least 120 steps so the run crosses the 100-step horizon; the terminal
re-emit and the `final_observation` aliasing check only fire there.
