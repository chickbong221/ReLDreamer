# TEEMO simulation probe

Build a fixed-size manipulation graph from MS-HAB segmentation + physics.
MS-HAB is read-only; all integration lives under `teemo_sim_probe/adapters/`.

> **Frozen.** This package is the predecessor pipeline, kept runnable as a
> self-contained demo: one camera, mean masked depth per node, target-conditioned
> pooling. The online pipeline the world model trains against moved to
> [`scenegraph/`](../scenegraph/README.md) — two cameras, frozen DINOv2
> appearance, no target conditioning.
>
> Still live for everyone: the rollout collector, both mining tools, and the
> mined assets under `configs/`. `scenegraph/` reads those and does not import
> anything else from here.

## Pipeline at a glance

```text
  ─── offline (run once per asset bump) ─────────────────────────────
                                                                      
   MS-HAB env  ──►  FetchCollectContactDataWrapper  ──►  <obj>.pkl    
                                                          │           
                          ┌───────────────────────────────┤           
                          ▼                               ▼           
                  build_affordances              build_subtask_whitelists
                          │                               │           
                          ▼                               ▼           
                  affordances.json              whitelists/*.json     
                                                                      
  ─── online (every simulator step) ─────────────────────────────────
                                                                      
   MS-HAB env  ──►  GraphBuilder ◄── affordances.json + whitelist     
                          │                                           
                          ▼                                           
                  Graph(nodes, edges)                                 
```

1. **Collect.** The wrapper buffers per-env rollouts; on success it commits
   one `<obj>.pkl` with poses, ee--obj / obj--obj contacts, supports, and raw
   bin samples.
2. **Mine affordances.** Per canonical object, derive grasp / contact /
   support / bottom / contain / key components.
3. **Mine whitelists.** Per `(subtask, target)`, derive members,
   interaction-type tokens, and bin edges.
4. **Run probe.** Each step the runtime gates nodes via the whitelist,
   scores compatibilities against the affordance asset, and emits absolute
   and temporal edges.

## Graph contract

Two node types:

* `ee` -- end effector (tcp + finger1 + finger2 folded into one node), always
  vertex index 0.
* `object` -- every non-robot actor or articulation link.

The vertex set is append-only per episode: an index is handed out on first
sight and never reused or reordered, so a node that leaves the view keeps its
position, its last pose, and its retained appearance. Each node carries
`(feat, kappa, visible)` where `feat` is the mean masked depth per camera in
metres, retained per camera while unseen.

A node appears only if (a) an ee link touched it during a successful demo
and (b) it is listed in the active per-`(subtask, target)` whitelist.

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
is the 0.9 quantile across all demo samples for the same `(subtask, target)`.
Compatibility absolute edges are fixed at `[1/3, 2/3]` (score is already in
`[0, 1]`). `configs/thresholds.yaml` provides fallbacks for relations the
asset omits.

## Asset shapes

Rollout pickle (schema v6, `<obj>.pkl`):

```text
{
  obj_id, entity_key, subtask_type, temporal_k,
  robot_qpos, obj_pose_wrt_base, tcp_pose_wrt_base,
  interaction_rollouts: [{
    target_key,
    interacted:   [{key, name, kind, max_ee_force, grasped?}],
    supports:     [{supporter, supported_key, force, dz, evidence,
                    supporter_pose, supported_pose, force_vector}],
    obj_contacts: [{a_key, b_key, a_pose, b_pose, force_vector, force}],
    bin_samples:  {<relation>: [floats]}
  }]
}
```

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

## End-to-end sweep: `set_table` across `pick` / `open` / `close`

The collector writes one pkl per `(subtask, obj_id)` at
`$MS_ASSET_DIR/data/robot_success_states/fetch/<subtask>/<obj_id>.pkl`.
Because the path only carries `obj_id`, the same `obj_id` shared across
tasks would overwrite (e.g. `024_bowl` exists in both `prepare_groceries`
and `set_table`) — always scope with `--task` when sweeping.

```bash
export MS_ASSET_DIR=/root/.maniskill
STATES_DIR="$MS_ASSET_DIR/data/robot_success_states"

# 0. One-time checkpoint download (skip if mshab_checkpoints/ already populated).
huggingface-cli download arth-shukla/mshab_checkpoints \
    --local-dir mshab_checkpoints

# 1. Collect successes for every set_table subtask.
#    --task set_table pins the task so shared obj_ids (024_bowl,
#    kitchen_counter, ...) don't collide with other tasks' pkls.
#    --num-envs 8 uses the multi-env force-query fix in
#    ``FetchCollectContactDataWrapper._pairwise_force`` -- lower this
#    only if GPU memory is tight, correctness is unaffected either way.
for SUB in pick open close; do
    python -m teemo_sim_probe.tools.collect_robot_success_states \
        --ckpt-root mshab_checkpoints/rl \
        --task set_table --subtask "$SUB" \
        --n-success 30 --num-envs 8 --no-skip-done
done

# 2. Mine affordances (one asset covers all subtasks; the miner walks
#    every subtask directory under --success-states-dir).
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$STATES_DIR" \
    --robot fetch --subtask pick \
    --out teemo_sim_probe/configs/affordances.json

# 3. Mine whitelists (one JSON per (subtask, target); the miner emits every
#    pkl it finds, so this covers pick + open + close in one call).
python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$STATES_DIR" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists

# 4. Run the probe on any set_table checkpoint you want to inspect.
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 2 --video

# 5. Tests.
python -m unittest discover teemo_sim_probe/tests
```

### Verify each subtask end-to-end

Between steps 1 and 3, use `verify_pipeline` to confirm that a specific
target's pkl carries the supporter you expect AND that the mined whitelist
JSON propagated it correctly. Pattern: pick one representative target per
subtask and one supporter you know is physically present in the task.

```bash
# pick: 024_bowl should show drawer3 or kitchen_counter body as supporter
python -m teemo_sim_probe.tools.verify_pipeline \
    --pkl "$STATES_DIR/fetch/pick/024_bowl.pkl" \
    --whitelist-dir teemo_sim_probe/configs/subtask_whitelists \
    --subtask pick --obj 024_bowl \
    --expect-key link:kitchen_counter-0/drawer3

# open: the kitchen_counter drawer articulation should be interacted
python -m teemo_sim_probe.tools.verify_pipeline \
    --pkl "$STATES_DIR/fetch/open/kitchen_counter.pkl" \
    --whitelist-dir teemo_sim_probe/configs/subtask_whitelists \
    --subtask open --obj kitchen_counter \
    --expect-key link:kitchen_counter-0/drawer3

# close: fridge door interaction
python -m teemo_sim_probe.tools.verify_pipeline \
    --pkl "$STATES_DIR/fetch/close/fridge.pkl" \
    --whitelist-dir teemo_sim_probe/configs/subtask_whitelists \
    --subtask close --obj fridge \
    --expect-key link:fridge-0/body
```

Each run prints three sections: pkl audit, obj_contacts A-vs-B triage,
whitelist JSON audit. The last line is a labeled verdict. If any subtask's
verdict is not "CORRECT", stop and inspect that pkl before proceeding to
step 3 — a schema mismatch or empty rollout list will otherwise silently
produce a broken whitelist JSON.

### Iterating on a single target

If you want to re-run for one object without redoing the entire sweep,
scope both the collector and the audit with `--obj`. The whitelist miner
always rescans the full directory, so it will re-emit that target's JSON
alongside the untouched ones.

```bash
# Overwrite one pkl only, then re-audit + re-mine + re-run the probe.
rm -f "$STATES_DIR/fetch/pick/024_bowl.pkl"

python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root mshab_checkpoints/rl \
    --task set_table --subtask pick --obj 024_bowl \
    --n-success 30 --num-envs 8 --no-skip-done

python -m teemo_sim_probe.tools.diagnose_bowl_supporter --skip-live

python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$STATES_DIR" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists

python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 2 --video
```

### Raw RGB rollout (no segmentation)

Use `--rgb-only` to run the pretrained checkpoint without requesting a
segmentation texture or building graphs. The mode discovers every RGB sensor
exposed by the environment and saves one PNG per camera per step. It defaults
to 200 steps, so each camera directory contains `frame_0000.png` through
`frame_0199.png`:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --rgb-only --out teemo_sim_probe/outputs/rgb_rollout
```

For the standard Fetch setup this produces `fetch_head/` and `fetch_hand/`,
each with 200 raw RGB PNGs. Pass `--steps N` to request a different count.

After a schema bump (currently rollout `v6` / whitelist `v4` / affordances
`v3`), re-run steps 1 → 2 → 3 with `--no-skip-done`. The runtime fails loud
at episode start when no matching whitelist exists for `(subtask, target)`.
