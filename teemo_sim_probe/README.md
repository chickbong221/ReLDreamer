# TEEMO simulation probe

The probe builds a fixed-size manipulation graph from simulator segmentation
and physics. MS-HAB is read-only; all integration and collection behavior
lives under `teemo_sim_probe/adapters/`.

## Graph contract

Two node types:

* `ee` — the end effector (tcp + finger1 + finger2 folded into one node).
* `object` — every non-robot actor or articulation link.

Handles have no special admission rule. A node appears only when an ee link
touched it during a successful offline rollout and it is present in the
active per-(subtask, target) whitelist.

### Relation vocabulary

Relations are grouped into three families. The visualizer paints chips in one
background color per family so the viewer can read `event` vs `spatial` vs
`affordance` at a glance.

| Family | Absolute relations | Absolute labels | Temporal relations | Temporal labels |
|---|---|---|---|---|
| **Event** | `contact`, `grasp`, `support` | `<relation>` / `no-<relation>` | `*-transition` | `gain-` / `lose-` / `maintain-` |
| **Spatial** | `planar-distance` | `near`, `medium`, `far` | `planar-distance-change` | `approaching-{slow,fast}`, `stable-distance`, `receding-{slow,fast}` |
|  | `height-offset` | `below`, `level`, `above` | `height-offset-change` | `lowering-{slow,fast}`, `stable-height`, `rising-{slow,fast}` |
| **Affordance** | `grasp-compatibility` | `match`, `partial-match`, `poor-match` | `grasp-compatibility-change` | `grasp-fit-{better,worse}-{slow,fast}`, `stable-grasp-fit` |
|  | `contact-compatibility` | `match`, `partial-match`, `poor-match` | `contact-compatibility-change` | `contact-fit-{better,worse}-{slow,fast}`, `stable-contact-fit` |

### Gating rules

1. **Spatial is always object-center.** Both `planar-distance` and
   `height-offset` use the object's center pose, never an affordance anchor.
2. **Affordance compatibility is `near`-only.** If the current
   `planar-distance` label is anything but `near`, no compatibility edge is
   emitted (selection cache stays warm so it does not churn).
3. **Whitelist governs compatibility per object.** `grasp-compatibility` is
   emitted only when the whitelist records `grasp` for that object;
   `contact-compatibility` only when it records `contact`.
4. **Grasp masks contact.** While the object is grasped, `contact`,
   `contact-transition`, `contact-compatibility`, and
   `contact-compatibility-change` are masked.
5. **Direct supporters are first-class.** `object/object` `support` edges are
   always emitted between admitted supporter and supported nodes.

### Compatibility scoring

For each object's active affordance component, the scorer measures
three mismatches against the live gripper-object configuration:

* `pos_mismatch` — tcp → anchor distance (metres)
* `orient_mismatch` — angle between tcp approach axis and component approach
  direction (radians)
* `width_mismatch` — `|qpos_sum − preferred_width|` (metres)

Each is clipped to `[0, 1]` by its `compat_norm` divisor, then the unweighted
mean is binned with `[1/3, 2/3]` into `match` / `partial-match` / `poor-match`.
`grasp-compatibility` uses all three components; `contact-compatibility` drops
the width term.

### Bin edges from demos

`planar-distance`, `height-offset`, and every `*-change` relation are binned
using **equal-width** splits of `[0, max]` (unsigned) or `[-max, max]`
(signed), where `max` is the per-relation maximum observed across successful
demonstrations of the same `(subtask, target)`. The collector samples those
maxes; the miner aggregates and writes them as `bin_edges` into the whitelist
asset; the runtime reads them at whitelist bind. `configs/thresholds.yaml`
provides fallback bins only for relations the asset omits.

## Offline pipeline

`FetchCollectContactDataWrapper` buffers each vector environment
independently and commits one schema-v5 record per successful rollout:

```text
{
  _schema_version: 5,
  obj_id, entity_key, subtask_type, temporal_k,
  robot_qpos, obj_pose_wrt_base, tcp_pose_wrt_base,
  interaction_rollouts: [{
    target_key,
    interacted: [{key, name, kind, max_ee_force, grasped?}],
    supports:   [...],
    bin_stats:  {planar_distance, height_offset,
                 planar_distance_change, height_offset_change}
  }],
  bin_stats: {...}     # aggregated across rollouts
}
```

Key behaviors:

* **Contact evidence is ee-only.** Only `tcp`, `finger1_link`, `finger2_link`
  contacts count toward whitelist membership. Robot-body bumps are ignored.
* **Interaction types are tracked.** A member that an ee link merely touched
  carries `interaction_types: ["contact"]`; one that the grasp predicate
  fired on carries `["contact", "grasp"]`.
* **One-hop supporters.** Direct supporters of an interacted entity are
  admitted; recursive closure is rejected.
* **Bin stats are sampled continuously.** Per `observe_stride` ticks the
  wrapper updates per-rollout running maxes of spatial values and their
  K-window absolute changes.

`tools/build_subtask_whitelists.py` aggregates per `(subtask, target)` into:

```text
{
  _schema_version: 3,
  subtask, target,
  members: { <key>: {roles, interaction_types, kind, name, ...} },
  bin_edges: { <relation>: [edges...] },
  compat_norm: {pos, orient, width}
}
```

`tools/build_affordances.py` consumes the pose arrays from the same pickles
and emits multi-modal `{anchor, approach_dir, width}` components per
canonical object key.

## Runtime pipeline

```text
active (subtask, target key)
  -> load per-(subtask, target) whitelist (binds bin_edges + interaction_types into cfg)
  -> build all non-robot segmentation candidates
  -> hard whitelist gate
  -> classify by affordance + whitelist role
  -> role-aware capacity (interacted > support > other)
  -> stable slots
  -> absolute edges:
       ee_object_spatial_event_edges   (every object, center-based)
       ee_object_compatibility_edges   (near-gated, whitelist-gated)
       object_object_edges             (contact / support, supporter-> supported)
  -> temporal edges (signed change + binary transitions over K frames)
```

There is no invisible active-target injection and no local-contact admission
path. The active target id selects an asset; it does not create a node.

## End-to-end commands

```bash
# 0. Point at the ManiSkill asset root (parent of data/).
export MS_ASSET_DIR=/root/.maniskill

# 1. Collect schema-v5 successful rollouts (one .pkl per object).
#    --no-skip-done is required when replacing older schemas.
python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root mshab_checkpoints/rl \
    --n-success 30 --num-envs 8 \
    --no-skip-done

# 2. Mine the affordance asset (multi-modal anchor/approach/width per object).
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --robot fetch --subtask pick \
    --out teemo_sim_probe/configs/affordances.json

# 3. Mine the per-(subtask, target) whitelists with interaction_types + bin_edges.
python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists

# 4. Run the MS-HAB probe and save overlays / graphs / (optional) video.
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 2 --video

# 5. Run the unit tests.
python -m unittest discover teemo_sim_probe/tests
```

Notes:

* `--save-every` controls rendering only; selection and temporal state
  advance every simulator step.
* `--whitelist-dir <path>` overrides the default whitelist location; useful
  when iterating on freshly mined assets.
* Re-running step 1 after a schema bump requires `--no-skip-done` so already
  saved pickles get overwritten with the new schema.

## Re-mining after a schema bump

Whenever this README's schema versions advance (currently rollout `v5` /
whitelist `v3`), every offline asset must be regenerated:

```bash
# (Optional) clear stale pickles + assets first.
rm -rf "$MS_ASSET_DIR/data/robot_success_states/fetch"
rm -f  teemo_sim_probe/configs/affordances.json
rm -rf teemo_sim_probe/configs/subtask_whitelists

# Then re-run steps 1 -> 2 -> 3 above.
```

The runtime fails loud at episode start when no matching whitelist exists for
`(subtask, target)`, so missing assets are caught immediately rather than
producing silent "admit everything" behavior.
