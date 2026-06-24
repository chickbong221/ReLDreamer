# teemo_sim_probe

Standalone probe that extracts a fixed-size **manipulation semantic graph**
(R1-R16) from simulator episodes and renders it as mask overlays + node-link
diagrams. Designed for MS-HAB (room-scale Fetch) and plain ManiSkill
(tabletop). Independent of `dreamerv3/` and `embodied/`.

```
seg ids
 -> drop background (seg id 0)
 -> merge robot gripper/fingers/TCP into ee
 -> drop helper/goal/marker + room/clutter            (--include-* to keep)
 -> classify pair types (uses affordance set)
 -> merge_persistent: k_persist=5 identity-keyed retention
 -> tau_i update (steps_since_seen)
 -> expand_local_contact: V_{t-1} one-hop, mask-gated
 -> dedup by live entity (collapse duplicates)
 -> apply_whitelist: HARD per-subtask gate                  (REQUIRED asset)
 -> overflow_truncate: keep n_slots=10 nearest to ee, node_id tiebreak
 -> identity-keyed slot assignment + reset_flag
 -> build absolute + temporal edges (valid slots only)
```

Every frame yields exactly `1 ee + N=10 object slots`; empty slots are padded
(`valid_mask=False`) so the downstream encoder sees fixed capacity.

## Relation vocabulary

Absolute (per frame):

| valid pair          | relation              | absolute labels                                                                                |
|---------------------|-----------------------|-------------------------------------------------------------------------------------------------|
| `ee--static_obj`    | `planar-distance`     | `very-near, near, medium, far, very-far`                                                        |
| `ee--static_obj`    | `height-offset`       | `far-below, below, level, above, far-above`                                                     |
| `ee--static_obj`    | `contact`             | `no-contact, contact`                                                                           |
| `ee--interactive_obj` | `planar-distance`   | `very-near, near, medium, far, very-far`                                                        |
| `ee--interactive_obj` | `height-offset`     | `far-below, below, level, above, far-above`                                                     |
| `ee--interactive_obj` | `orientation-alignment` | `aligned, near-aligned, partial, misaligned, very-misaligned`                              |
| `ee--interactive_obj` | `contact`           | `no-contact, contact`                                                                           |
| `ee--interactive_obj` | `grasp`             | `no-grasp, grasp`                                                                               |
| `object--object`    | `contact`             | `no-contact, contact`                                                                           |
| `object--object`    | `support`             | `no-support, support`                                                                           |

Temporal (over `t-K:t`, K=5):

| valid pair          | temporal relation             | temporal labels                                                                                |
|---------------------|-------------------------------|------------------------------------------------------------------------------------------------|
| `ee--*`             | `planar-distance-change`      | `approach-{slow,medium,fast}, stable-distance, recede-{slow,medium,fast}`                      |
| `ee--*`             | `height-offset-change`        | `move-up-{slow,medium,fast}, stable-height, move-down-{slow,medium,fast}`                      |
| `ee--interactive_obj` | `orientation-alignment-change` | `improve-alignment-{slow,medium,fast}, stable-alignment, worsen-alignment-{slow,medium,fast}` |
| `ee--*`             | `contact-transition`          | `gain-contact, lose-contact, maintain-contact, (maintain-no-contact bg only)`                  |
| `ee--interactive_obj` | `grasp-transition`          | `gain-grasp, lose-grasp, maintain-grasp, (maintain-no-grasp bg only)`                          |
| `object--object`    | `contact-transition`          | `gain-contact, lose-contact, maintain-contact, (maintain-no-contact bg only)`                  |
| `object--object`    | `support-transition`          | `gain-support, lose-support, maintain-support, (maintain-no-support bg only)`                  |

Spatial reference for `ee--interactive_obj` (`planar-distance`,
`height-offset`): selected affordance anchor when a mined component exists,
else object center. The choice is recorded per-node as
`attributes['spatial_ref']` for audit. `object--object` is mutually exclusive
per pair (true `support` suppresses the `contact` edge for that pair).

## Adapter

`adapters/privileged_state.py` gathers what the simulator already exposes (no
`env.get_privileged_state` exists):

| need                | api                                                            |
|---------------------|----------------------------------------------------------------|
| seg -> entity       | `env.unwrapped.segmentation_id_map`                            |
| ee pose             | `agent.tcp_pose` / `agent.tcp.pose`                            |
| gripper width       | `agent.robot.qpos[-2:].sum()` (Fetch qpos-sum convention)      |
| contact / grasp     | `scene.get_pairwise_contact_forces(a,b)`, `agent.is_grasping`  |
| MS-HAB active task  | `subtask_objs`, `subtask_articulations`, `task_plan`, `subtask_pointer` |
| original obj_id     | `env.build_config_idx_to_task_plans[bci][tpi].subtasks[ptr]`   |

A new backend only needs a sibling adapter.

## Required setup (one-time per benchmark)

Mined assets required at startup. `load_config` raises `FileNotFoundError`
if the affordance asset is missing or empty; the per-subtask whitelist is
resolved lazily at each episode reset and raises if no matching file is
present (Track A: fail-loud, no silent fallback to "admit everything"):

| asset                                                       | what it gates                    | produced by |
|-------------------------------------------------------------|----------------------------------|-------------|
| `teemo_sim_probe/configs/affordances.json`                  | relation vocabulary + anchors    | step 3      |
| `teemo_sim_probe/configs/subtask_whitelists/*.json`         | per-(subtask, target) node gate  | step 4      |

Run the four steps below in order, top to bottom. Then go to **Run the probe**.

### 0. Set the asset dir

```bash
export MS_ASSET_DIR=/root/.maniskill
```

All later steps read this. **MS_ASSET_DIR is the ManiSkill root (parent of
`data/`), not the data dir itself** -- ManiSkill internally resolves
`$MS_ASSET_DIR/data` as `mani_skill.ASSET_DIR`, so a trailing `/data` would
cause a doubled `data/data/` segment when ManiSkill loads the ReplicaCAD scene.
Set this explicitly to whatever location holds your YCB + ReplicaCAD bundles
(i.e. the directory whose `data/` subfolder contains `scene_datasets/`).

### 1. Download released checkpoints

```bash
huggingface-cli download arth-shukla/mshab_checkpoints \
    --local-dir mshab_checkpoints
```

Checkpoints land at `mshab_checkpoints/rl/<task>/pick/<obj_id>/policy.pt`.

### 2. Populate `$MS_ASSET_DIR/data/robot_success_states/`

```bash
python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root mshab_checkpoints/rl \
    --n-success 30 --num-envs 8
```

Rolls each per-object Fetch pick policy out under
`teemo_sim_probe.adapters.collect_contact_data.FetchCollectContactDataWrapper`
(a local sibling adapter -- `mshab/` is untouched) and writes one pickle per
YCB id to `$MS_ASSET_DIR/data/robot_success_states/fetch/pick/<obj_id>.pkl`.

Schema (`_schema_version: 2`, strict superset of the upstream wrapper):
`{obj_id, subtask_type, robot_qpos[N x 15], obj_pose_wrt_base[N x 7],
contact_graphs[N]}`. `build_affordances.py` (step 3) reads only the original
fields. `build_subtask_whitelists.py` (step 4) reads `contact_graphs`.

Re-runnable: pickles that already have `>= --n-success` rows are skipped.

Useful flags:

| flag                    | effect                                                              |
|-------------------------|---------------------------------------------------------------------|
| `--obj 024_bowl`        | only collect this YCB id (repeatable)                               |
| `--task tidy_house`     | only collect from this task tree (repeatable)                       |
| `--num-envs 8`          | parallel envs per object (GPU sim)                                  |
| `--n-success 30`        | target successes per object before stopping                         |
| `--max-total-steps N`   | hard cap on env steps per object (safety)                           |
| `--stall-steps N`       | abandon an object if no new success in N steps                      |
| `--no-skip-done`        | re-collect even if the `.pkl` already has enough samples            |

Pick only. Place success requires `~is_grasped` at `ee_rest_world_pose`, so
`inv(obj_pose_wrt_base) * tcp_in_base` from a place rollout would learn the
robot's rest pose rather than an object affordance.

### 3. Mine the affordance set (R6)

```bash
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --robot fetch --subtask pick \
    --out teemo_sim_probe/configs/affordances.json \
    --n-components 4
```

FK is required; the miner aborts rather than write `[0,0,0]` placeholders.

### 4. Mine per-subtask whitelists (Track A)

```bash
python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists
```

Walks the success-state pkls, BFS-closes from the target over the contact
pair graph (default `--max-hops 2`), and writes one JSON per
`(subtask, canonical_target)`, e.g. `pick_024_bowl.json`,
`open_drawer3.json`. Tune `--min-support-frac` (default 0.3, lenient) and
`--min-contact-frac` (default 0.6, strict) if the closures look too thin or
too noisy.

### Shortcut: steps 2-4 in one call

After steps 0 and 1, `setup_assets.sh` runs 2, 3, 4 in order with the same
flags shown above:

```bash
bash teemo_sim_probe/tools/setup_assets.sh mshab_checkpoints 30 4
#                                          ^ckpt root  ^N succ  ^K affordance components
```

Use either path -- the explicit four steps or this wrapper -- not both.

## Run the probe

After the four setup steps above:

Plain ManiSkill:

```bash
python -m teemo_sim_probe.run_ms_probe --env-id PickCube-v1 --steps 50 --video
```

MS-HAB with a released checkpoint:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/tidy_house/pick/all \
    --steps 60 --video
```

Per-object SAC checkpoint -- runner infers the task plan from the path:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 20 --width 128 --height 128 --video
```

MS-HAB names merged subtask targets `obj_0`. By default the probe resolves
this to the actual actor name (`env-0_024_bowl-3`); pass
`--mshab-object-name merged` to preserve the internal name.

If `policy.pt` is missing the runner falls back to random actions.

## Ablation flags

Same flags on both run scripts:

| flag                      | effect                                                |
|---------------------------|-------------------------------------------------------|
| `--k-persist N`           | override persistence window (frames)                  |
| `--n-slots N`             | override n_slots=10 capacity                          |
| `--no-local-contact`      | disable the V_{t-1} one-hop expansion                 |
| `--whitelist-dir <path>`  | override the per-subtask whitelist directory          |

## Tests

```
python -m unittest discover teemo_sim_probe/tests
```

Covers schema serialization, affordance lookup / transforms, temporal anchor
reset, relation rules, whitelist gate + link specificity, overflow
truncation, persistence window survival, and slot identity / reset_flag.
