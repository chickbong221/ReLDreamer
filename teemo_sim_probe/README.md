# TEEMO simulation probe

Build a fixed-size manipulation graph from MS-HAB segmentation + physics.
MS-HAB is read-only; all integration lives under `teemo_sim_probe/adapters/`.

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

* `ee` -- end effector (tcp + finger1 + finger2 folded into one node).
* `object` -- every non-robot actor or articulation link.

A node appears only if (a) an ee link touched it during a successful demo
and (b) it is listed in the active per-`(subtask, target)` whitelist.

### Relation vocabulary

| Family | Pair type | Relation | Labels |
|---|---|---|---|
| **Physical state** | ee--obj | `grasp` | `grasp` |
|  | ee--obj, obj--obj | `contact` | `contact` |
|  | obj--obj | `support` | `support` |
|  | obj--obj | `contain` | `contain` |
| **Spatial** | ee--obj | `planar-distance` | `near` / `medium` / `far` |
|  | ee--obj | `height-offset` | `below` / `level` / `above` |
| **Affordance** | ee--obj | `grasp-compatibility` | `match` / `partial-match` / `poor-match` |
|  | ee--obj, obj--obj | `contact-compatibility` | (same) |
|  | obj--obj | `support-compatibility` | (same) |
|  | obj--obj | `contain-compatibility` | (same) |

Every spatial and affordance relation has a `*-change` sibling, binned over
a `K`-frame window into a 5-way signed label (`*-fast`, `*-slow`, stable,
opposite slow/fast). Physical-state edges have no transitions: consecutive
absolute frames are sufficient.

### Gating rules

1. **One physical-state edge per pair.** ee--obj: `grasp`, else `contact`.
   obj--obj: strict priority `contain > support > contact`.
2. **Spatial is object-center**, computed only for `ee--obj`.
3. **Affordance compatibility is `near`-only.** If the endpoint centers do
   not bin to `near`, no compat edge is emitted.
4. **Whitelist gates compatibility per object.** A compat edge fires only
   when both endpoints' `interaction_types` carry the matching token
   (`contact` / `grasp` / `support` / `contain`).
5. **Contact-compat is masked under grasp.** The edge still emits with
   `masked=True` and `suppressed_by_grasp=True` so the temporal buffer drops
   its history; the parallel physical-state `contact` edge is not emitted.

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

After a schema bump (currently rollout `v6` / whitelist `v4` / affordances
`v3`), re-run steps 1 → 2 → 3 with `--no-skip-done`. The runtime fails loud
at episode start when no matching whitelist exists for `(subtask, target)`.
