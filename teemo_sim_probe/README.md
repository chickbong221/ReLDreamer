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
background color per family so the viewer can read `physical_state` vs
`spatial` vs `affordance` at a glance.

| Family | Pair type | Absolute relation | Absolute labels | Change relation | Change labels |
|---|---|---|---|---|---|
| **Physical state** | `actor--obj`, `obj--obj` | `contact` | `contact` | — | — |
|  | `actor--obj` | `grasp` | `grasp` | — | — |
|  | `obj--obj` | `support` | `support` | — | — |
|  | `obj--obj` | `contain` | `contain` | — | — |
| **Spatial** | `actor--obj` | `planar-distance` | `far`, `medium`, `near` | `planar-distance-change` | `approaching-{slow,fast}`, `stable-distance`, `receding-{slow,fast}` |
|  | `actor--obj` | `height-offset` | `below`, `level`, `above` | `height-offset-change` | `lowering-{slow,fast}`, `stable-height`, `rising-{slow,fast}` |
| **Affordance** | `actor--obj`, `obj--obj` | `contact-compatibility` | `poor-match`, `partial-match`, `match` | `contact-compatibility-change` | `contact-fit-{better,worse}-{slow,fast}`, `stable-contact-fit` |
|  | `actor--near_obj` | `grasp-compatibility` | `poor-match`, `partial-match`, `match` | `grasp-compatibility-change` | `grasp-fit-{better,worse}-{slow,fast}`, `stable-grasp-fit` |
|  | `obj--near_obj` | `support-compatibility` | `poor-match`, `partial-match`, `match` | `support-compatibility-change` | `support-fit-{better,worse}-{slow,fast}`, `stable-support-fit` |
|  | `obj--near_obj` | `contain-compatibility` | `poor-match`, `partial-match`, `match` | `contain-compatibility-change` | `contain-fit-{better,worse}-{slow,fast}`, `stable-contain-fit` |

Physical-state transitions are NOT separately annotated: consecutive absolute
frames are sufficient evidence of their dynamics. The single positive label
form (`contact`, not `contact`/`no-contact`) reflects that an edge either
exists or doesn't.

### Gating rules

1. **Spatial is always object-center.** Both `planar-distance` and
   `height-offset` use the object's center pose, never an affordance anchor.
   Spatial is emitted only for the `actor--obj` pair type; obj-obj near-gating
   is computed internally for compatibility edges.
2. **Affordance compatibility is `near`-only.** If the current
   `planar-distance` between the two endpoint centers is anything but `near`,
   no compatibility edge is emitted (selection cache stays warm so it does
   not churn).
3. **Whitelist governs compatibility per object.** Each whitelist member
   carries an `interaction_types` set drawn from `{contact, grasp, support,
   contain}`. A compatibility edge fires only when both endpoints' types
   include the matching token.
4. **One physical-state edge per pair.** ee--object emits `grasp` when the
   grasp predicate fires, else `contact` when in contact -- never both.
   obj--obj uses strict priority `contain > support > contact`: a pair that
   is geometrically contained gets only a `contain` edge, a non-contained
   pair with vertical-dominated support force gets only a `support` edge,
   and any remaining contacting pair gets a `contact` edge. The
   `contact-compatibility` edge still carries
   `attributes['suppressed_by_grasp']=True` (and `masked=True`) when an
   endpoint is grasped, which the temporal buffer uses to drop that edge's
   history; the physical-state contact edge itself is no longer emitted in
   that case.
5. **Direct supporters are first-class.** `obj--obj` `support` edges are
   emitted between admitted supporter and supported nodes (vertical force
   ratio + center dz) for pairs that did not already qualify for `contain`.
   `contain` is decided geometrically: transform the containee's mined key
   point into the container's frame, then check that it lies within
   `[0, depth]` along the entry axis and within `opening_radius` transverse
   -- the same template ManiSkill's `PegInsertionSide-v1.has_peg_inserted`
   uses.

### Compatibility scoring

All four compatibility relations score an unweighted mean of `[0, 1]` per-
component mismatches and bin with `[1/3, 2/3]` into `match` / `partial-match`
/ `poor-match`. Per-component normalizers live under `cfg["compat_norm"]`
(overridable per `(subtask, target)` in the whitelist asset). Components per
relation:

* **`grasp-compatibility`** (actor → near_obj):
  * `pos_mismatch` — tcp → anchor distance (metres, norm `pos`)
  * `orient_mismatch` — angle between tcp approach axis and component approach
    direction (radians, norm `orient`)
  * `width_mismatch` — `|qpos_sum − preferred_width|` (metres, norm `width`)
* **`contact-compatibility`** drops the width term:
  * actor-obj: `pos` + `orient` against the active grasp component.
  * obj-obj: `pos` between matched contact anchors + `orient` between
    each side's outward normal (anti-parallel expected at a real contact).
* **`support-compatibility`** (obj → near_obj, supporter → supported):
  * `xy_mismatch` — in-plane offset of supported.`bottom_anchor` from
    supporter.`surface_anchor`, clipped to 0 inside `footprint_radius`
    (norm `xy`)
  * `vertical_mismatch` — gap / interpenetration along surface normal
    (norm `vertical`)
  * `orient_mismatch` — angle between supported.`bottom_normal` and the
    inverted surface normal (norm `orient`)
* **`contain-compatibility`** (obj → near_obj, container → containee), modeled
  on PegInsertionSide:
  * `radial_mismatch` — perpendicular distance from containee.`key_anchor`
    to the entry axis line minus `opening_radius`, clipped at 0 (norm
    `radial`)
  * `axial_mismatch` — distance past the `[0, depth]` interval along the
    entry axis, clipped at 0 (norm `axial`)
  * `orient_mismatch` — angle between containee.`key_axis` and container.
    `entry_axis` (norm `orient`)

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
independently and commits one schema-v6 record per successful rollout:

```text
{
  _schema_version: 6,
  obj_id, entity_key, subtask_type, temporal_k,
  robot_qpos, obj_pose_wrt_base, tcp_pose_wrt_base,
  interaction_rollouts: [{
    target_key,
    interacted: [{key, name, kind, max_ee_force, grasped?}],
    supports:   [{supporter, supported_key, force, vertical_force_ratio,
                  dz, vertical_support, evidence: "force"|"geometric",
                  supporter_pose, supported_pose, force_vector}],
    obj_contacts: [{a_key, b_key, a_pose, b_pose, force_vector, force}],
    bin_stats:  {planar_distance, height_offset,
                 planar_distance_change, height_offset_change},  # per-rollout 0.95 quantile
    bin_samples: {<relation>: [floats]}                          # raw per-tick samples
  }],
  bin_stats: {...}     # max of per-rollout quantiles
}
```

Key behaviors:

* **Contact evidence is ee-only.** Only `tcp`, `finger1_link`, `finger2_link`
  contacts count toward whitelist membership. Robot-body bumps are ignored.
* **Interaction types are tracked.** A member that an ee link merely touched
  carries `interaction_types: ["contact"]`; one that the grasp predicate
  fired on carries `["contact", "grasp"]`.
* **One-hop supporters.** Direct supporters of an interacted entity are
  admitted; recursive closure is rejected. A geometric fallback (candidate
  with tight xy overlap to the supported entity, ranked by xy overlap first
  then |dz|) catches resting receptacles that PhysX GPU silences in pairwise
  force queries. The dz window is asymmetric (`-0.15 m ≤ dz ≤ +0.5 m`)
  because SAPIEN link frames sit at joint origins, so a real supporter's
  center can be slightly *above* the supported entity's center.
* **Articulation keys are scene-config-set agnostic.** ReplicaCAD tags
  articulations with `scs-[N]_` or `scs-[N,M]_` prefixes that vary per build
  config. The miner and runtime both strip that prefix when forming stable
  keys, so a whitelist member `link:kitchen_counter-0/drawer3` matches the
  same logical articulation across every scene that loads it.
* **Spatial sampling is subject-restricted.** Per `observe_stride` ticks the
  wrapper samples ee→object planar distance / height offset (and K-window
  changes) **only** against the target plus already-known interacted /
  supporter entities -- not against every non-robot actor in the scene --
  so walls and far cabinet links cannot pump the bin range.
* **Bin aggregation is robust.** The collector emits a per-rollout 0.95
  quantile in `bin_stats` and the raw per-tick samples in `bin_samples`; the
  miner uses the samples to compute a quantile across all rollouts, then
  applies a per-relation sanity ceiling. A single autoreset jump or physics
  blow-up can no longer pin the bin edges to absurd values.
* **Done-tick observation is skipped.** MS-HAB autoresets inside the wrapped
  `step()`, so observing a done env this tick would mix the *next* episode's
  pose with the *current* rollout's stats. The wrapper skips those envs and
  also drops the first few ticks after every reset.

`tools/build_subtask_whitelists.py` aggregates per `(subtask, target)` into:

```text
{
  _schema_version: 4,
  subtask, target,
  members: { <key>: {roles, interaction_types, kind, name, ...} },
  bin_edges: { <relation>: [edges...] },
  bin_stats_robust: { <relation>: value },   # quantile fed to derive_bin_edges
  bin_stats_observed: { <relation>: value }, # max across all samples (audit)
  compat_norm: {pos, orient, width, xy, vertical, radial, axial}
}
```

`interaction_types` per member is a subset of `{contact, grasp, support,
contain}`, each token controlling which compatibility edges runtime emits for
that object.

The miner takes a quantile (default 0.9) across all per-tick samples (or per-
rollout quantiles when only legacy `bin_stats` is available), then clamps the
result to a per-relation sanity ceiling. It also logs a warning when an
`interacted` target has zero supporters across all rollouts -- almost always a
signal that the receptacle was resting-contact-only and the collector's
geometric supporter fallback failed to fire.

`tools/build_affordances.py` consumes pose arrays + obj-obj contact / support
event samples from the same pickles and emits an `affordances.json` (schema
v3) with up to six per-relation component lists per canonical object:

* `grasp_components` — `{anchor, approach_dir, width}` (ee → object).
* `contact_components` — `{anchor, outward_normal}` (obj-obj contact).
* `support_components` — `{surface_anchor, surface_normal, footprint_radius}`
  on supporters.
* `bottom_components` — `{bottom_anchor, bottom_normal}` on supported objects.
* `contain_components` / `key_components` — PegInsertionSide-style entry +
  key descriptors. MS-HAB has no containment env, so these stay empty for
  MS-HAB pickles; collecting from PegInsertionSide-v1 would populate them.

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
       ee_object_spatial_event_edges        (every object, center-based)
       ee_object_compatibility_edges        (near-gated, whitelist-gated)
       object_object_edges                  (contact / support / contain)
       object_object_compatibility_edges    (contact / support / contain compat)
  -> temporal edges (signed change over K frames; no transitions)
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

Whenever this README's schema versions advance (currently rollout `v6` /
whitelist `v4` / affordances `v3`), every offline asset must be regenerated:

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
