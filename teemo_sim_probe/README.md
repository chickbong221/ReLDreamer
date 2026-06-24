# teemo_sim_probe

Standalone probe that extracts a fixed-size **manipulation semantic graph**
(R1-R16) from simulator episodes and renders it as mask overlays + node-link
diagrams. Designed for MS-HAB (room-scale Fetch) and plain ManiSkill
(tabletop). Independent of `dreamerv3/` and `embodied/`.

```
seg ids
 -> R1 drop background
 -> R2 merge robot gripper/fingers/TCP into ee
 -> R3 drop helper/goal/marker + room/clutter        (--include-* to keep)
 -> R4 keep only entities in E_domain                (REQUIRED at startup)
 -> R5 one-hop direct expansion                       (mined offline)
 -> R6 affordance-set decides relation vocabulary     (REQUIRED at startup)
 -> R7 local-contact exception                        (V_{t-1}, one hop)
 -> R8 k=5 frame persistence
 -> R9 drop decorative articulation parts
 -> R10 continuous rule-score
 -> R11 top N=10 with refresh quota N_refresh=2
 -> R12 identity-keyed slots + reset_flag
 -> R13 no oracle active-target forcing
 -> R14 invisible state-bearing kept only via one-hop
 -> R15 labels only for valid slots
 -> R16 privileged labels train-time only
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

Both mined assets are **required** at startup. `load_config` raises
`FileNotFoundError` if either is missing or empty so the filtering pipeline
cannot silently degrade to an unmined graph:

| asset                                      | rule it gates | produced by |
|--------------------------------------------|---------------|-------------|
| `teemo_sim_probe/configs/affordances.json` | R6            | step 3      |
| `teemo_sim_probe/configs/e_domain.json`    | R4            | step 4      |

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
`mshab.envs.wrappers.collect_data.FetchCollectRobotInitWrapper` and writes
one pickle per YCB id to
`$MS_ASSET_DIR/data/robot_success_states/fetch/pick/<obj_id>.pkl` -- the schema
`build_affordances.py` and `build_e_domain.py` expect:
`{obj_id, robot_qpos[N x 15], obj_pose_wrt_base[N x 7]}`. Re-runnable:
pickles that already have `>= --n-success` rows are skipped.

Skip this step if `$MS_ASSET_DIR/data/robot_success_states/fetch/pick/*.pkl`
already exists (the MS-HAB asset bundle sometimes ships these).

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

### 4. Mine E_domain (R4)

```bash
python -m teemo_sim_probe.tools.build_e_domain \
    --task-plans-dir "$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans" \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --splits train \
    --out teemo_sim_probe/configs/e_domain.json
```

Train-split only (R4 provenance).

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
    --ckpt-dir E:/Code/ReLDreamer/mshab_checkpoints/rl/tidy_house/pick/all \
    --steps 60 --video
```

Per-object SAC checkpoint -- runner infers the task plan from the path:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir E:/Code/ReLDreamer/mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 20 --width 128 --height 128 --video
```

MS-HAB names merged subtask targets `obj_0`. By default the probe resolves
this to the actual actor name (`env-0_024_bowl-3`); pass
`--mshab-object-name merged` to preserve the internal name.

If `policy.pt` is missing the runner falls back to random actions.

## Ablation flags

Same flags on both run scripts:

| flag                      | effect (paper row)              |
|---------------------------|---------------------------------|
| `--k-persist N`           | override R8 persistence window  |
| `--n-refresh N`           | override R11 refresh quota      |
| `--n-slots N`             | override N=10 slot capacity     |
| `--no-local-contact`      | disable R7                       |
| `--dist-only`             | zero all weights except `w_dist` |
| `--oracle-active-target`  | R13 oracle row                  |

## Tests

```
python -m unittest discover teemo_sim_probe/tests
```

Covers schema serialization, affordance lookup / transforms, temporal anchor
reset, relation rules, selector score, refresh quota, persistence window, and
slot identity / reset_flag.
