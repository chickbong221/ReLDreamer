# teemo_sim_probe

A standalone probe that extracts TEEMO **manipulation semantic graphs** from
simulator episodes and renders them as mask overlays + node-link diagrams.

It lives at the repo root, separate from `dreamerv3/` and `embodied/`, and
does not touch Dreamer.

The current focus is **ManiSkill-HAB (MS-HAB)** because it ships task-aware
target metadata, action checkpoints, and 10 mined-friendly YCB objects. The
pipeline is intentionally written against a small adapter (`adapters/
privileged_state.py`) so the same graph builder, eligibility classifier, and
affordance vocabulary can be reused on:

| Backend            | Status                                                |
|--------------------|-------------------------------------------------------|
| ManiSkill-HAB      | first-class (this README)                              |
| ManiSkill 3        | runs with random actions today; checkpoint slot in `run_ms_probe` |
| BEHAVIOR-1K        | planned -- add a sibling `privileged_state` adapter   |

When extending to a new backend you implement one adapter (segmentation map,
TCP pose, contact / grasp predicate, optional task-active-target hooks) and
optionally drop an `affordances.json` produced by a per-benchmark miner. Every
other layer is benchmark-agnostic.

## What the pipeline does per frame

1. Read the segmentation image and merge robot gripper / fingers into a single
   `ee` node (`core/node_builder.py`).
2. Build one `object` node per non-robot Actor / Link that survives the
   visible-area threshold; persistent task targets are kept even when occluded.
3. Classify each object node as `static_object` or `interactive_object`
   (`node_builder.classify_pair_types`) -- this drives the relation vocabulary.
4. Emit absolute edges per class (`core/relation_rules.py`).
5. Run the edges through `core/temporal_buffer.py` for K-frame signed-change /
   transition labels.
6. Persist as `graph_XXXX.json`, draw masks (`overlay_XXXX.png`) and node-link
   diagrams (`graph_XXXX.png`), and optionally muxed `probe.mp4`.

Two node types only, matching the draft: `ee` and `object`.

## Eligibility-based relation vocabulary

`node_builder.classify_pair_types` annotates each object with a `pair_type`.
The classifier uses (in priority order):

1. has a mined affordance set                 -> `interactive_object`
2. is the MS-HAB active handle link           -> `interactive_object`
3. name matches `handle/knob/lever/drawer`    -> `interactive_object`
4. name matches `table/wall/floor/...`        -> `static_object`
5. otherwise: free actors default interactive, links default static

Rule (1) is the primary "has-an-affordance-set" criterion (paper-aligned);
(2)-(5) are robust fallbacks for objects that have not been mined yet and for
benchmarks where no affordance asset exists at all.

The emitted edge vocabulary follows `pair_type`:

```
ee--static_object         planar-distance, height-offset, contact
                          (center-based)
ee--interactive_object    planar-distance, height-offset, orientation-alignment,
                          contact, grasp
                          (+ gripper-width-alignment sub-signal)
                          (anchor-based when an asset exists; falls back to
                           the object center otherwise -- orientation-alignment
                           is simply skipped without a mined direction)
object--object            contact, support  (mutually exclusive)
```

Temporal labels (`*-change` / `*-transition`) are produced automatically by
`temporal_buffer.py` over a K-frame horizon. Anchor-bound histories
(planar-distance, height-offset, orientation-alignment, gripper-width-
alignment for interactive objects) are reset on `a_star` switches and on edge
disappearance so signed-change labels never compare across components.

Thresholds live in `configs/thresholds.yaml` with separate `tabletop`
(ManiSkill) and `room_scale` (MS-HAB) profiles.

## The privileged-state adapter

Neither ManiSkill 3 nor MS-HAB ships a single "semantic graph state" function.
`adapters/privileged_state.py` implements that name by gathering primitives
that *do* exist:

| Need            | API                                                                 |
|-----------------|---------------------------------------------------------------------|
| seg -> entity   | `env.unwrapped.segmentation_id_map` (id 0 = background)             |
| ee pose         | `agent.tcp_pose` (Fetch) / `agent.tcp.pose` (Panda)                 |
| gripper width   | `agent.robot.qpos[-2:].sum()` (Fetch, qpos-sum convention)          |
| fingers         | `agent.finger1_link`, `agent.finger2_link`                          |
| contact         | `scene.get_pairwise_contact_forces(a, b)`                           |
| grasp           | `agent.is_grasping(obj, max_angle=30)` (MS-HAB convention)          |
| MS-HAB targets  | `subtask_objs`, `subtask_articulations`, `task_plan`, `subtask_pointer` |
| Original obj_id | `env.build_config_idx_to_task_plans[bci][tpi].subtasks[ptr].obj_id` |

MS-HAB caveats handled here: `subtask_objs[i]` / `subtask_articulations[i]`
can be `None` (close / navigate subtasks); handle links come from
`subtask_articulations[i].links[subtask.articulation_handle_link_idx]`;
`task_plan[ptr]` holds the **merged** name (`obj_0`), so the original
`024_bowl-3` is recovered via `build_config_idx_to_task_plans` for affordance
lookup. Goal actors live in `_hidden_objects` and are excluded by default.

A new backend implements the same adapter; everything downstream is shared.

## Affordance assets

The affordance asset is a single JSON file
(`configs/affordances.json`) loaded once at startup. The repo ships an
**empty** asset -- without mined data the runtime emits zero affordance edges
(no crashes, no placeholders).

### Schema

Per canonical object key (10 known objects in MS-HAB:
`002_master_chef_can`, `003_cracker_box`, `004_sugar_box`,
`005_tomato_soup_can`, `007_tuna_fish_can`, `008_pudding_box`,
`009_gelatin_box`, `010_potted_meat_can`, `013_apple`, `024_bowl`):

```json
{
  "_schema_version": 2,
  "objects": {
    "024_bowl": {
      "raw_obj_id": "024_bowl",
      "n_samples": 1234,
      "components": [
        { "anchor": [0.000, 0.000, 0.020],
          "approach_dir": [0.0, 0.0, 1.0],
          "width": 0.045,
          "n_support": 412 },
        { "anchor": [0.030, 0.000, 0.010],
          "approach_dir": [0.0, 0.0, 1.0],
          "width": 0.050,
          "n_support": 397 }
      ]
    }
  }
}
```

Each component carries three pieces of pose-invariant information:

- **`anchor`** -- 3D point in the OBJECT frame (metres). "Where on the bowl
  the gripper grips."
- **`approach_dir`** -- OBJECT-frame unit vector giving the gripper approach
  axis at success. Optional; schema-v1 assets without it still load (the
  runtime simply skips `orientation-alignment` for that component).
- **`width`** -- gripper qpos-sum at success
  (`robot_qpos[-2] + robot_qpos[-1]` for Fetch). The runtime uses the same
  convention (`adapters/privileged_state.compute_gripper_width`) so the two
  match exactly. Do not mix in the URDF finger-origin offset of `+0.03085 m`
  -- it would systematically shift every runtime reading relative to the
  mined value.

At runtime, every frame we transform each component's `anchor` and
`approach_dir` through the **current** `obj_pose_world`, pick
`a_star = argmin(TCP -> anchor)`, then emit the eligible anchor-based edges.

### Collecting an affordance set

The repo ships an extensible miner at `tools/build_affordances.py`. It is the
**only** benchmark-specific piece of the affordance pipeline; the on-disk
schema, the runtime lookup (`core/affordance.py`), and the eligibility
classifier are reused across backends.

The current miner reads MS-HAB pick-success rollouts saved by
`mshab.envs.wrappers.collect_data.FetchCollectRobotInitWrapper`. To support
another backend (vanilla ManiSkill, BEHAVIOR-1K, ...), drop a sibling miner
that emits the same JSON shape and reuses
`teemo_sim_probe.core.affordance.canonical_affordance_key`. Per-benchmark
checkpoints can then be plugged in without touching anything downstream.

**About the two MS-HAB data artifacts (don't confuse them):**

- `rearrange-dataset/<task>/<subtask>/<obj_id>.h5` -- the *IL training
  dataset*, produced by MS-HAB's "Efficient, Controlled Data Generation at
  Scale" pipeline. Full trajectories, rule-based event-labeled and filtered.
  **Not what this miner consumes.**
- `robot_success_states/<robot>/<subtask>/<obj_id>.pkl` -- the *robot init /
  success-state data*, produced by `FetchCollectRobotInitWrapper`. Just
  `(robot_qpos, obj_pose_wrt_base)` at success moments, one object per file.
  **This is what the miner reads.**

**Step 1 -- Download MS-HAB's released checkpoints.** Training from scratch
is not required; the released RL checkpoints already cover all 10 YCB objects:

```bash
huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints
```

Checkpoints live at `mshab_checkpoints/rl/<task>/pick/<obj_id>/policy.pt`.

**Step 2 -- Obtain `robot_success_states/`.** Check `$MS_ASSET_DIR/` first --
this directory may already ship as part of the MS-HAB asset bundle (it backs
`spawn_data.pt`). If missing, collect it by running
`FetchCollectRobotInitWrapper` over one rollout per YCB id with its matching
checkpoint (the wrapper asserts single-object task plans):

```python
# sketch: collect_success_states.py
import gymnasium as gym
from mani_skill import ASSET_DIR
from mshab.envs.make import make_env
from mshab.envs.wrappers.collect_data import FetchCollectRobotInitWrapper

for obj_id in ["002_master_chef_can", "003_cracker_box", "004_sugar_box",
               "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box",
               "009_gelatin_box", "010_potted_meat_can", "013_apple",
               "024_bowl"]:
    env = make_env(...,
        task="tidy_house",   # or set_table for 013_apple / 024_bowl variants
        subtask="pick",
        split="train",
        task_plan_fp=f".../task_plans/<task>/pick/train/{obj_id}.json",
        spawn_data_fp=f".../spawn_data/<task>/pick/train/spawn_data.pt",
    )
    env = FetchCollectRobotInitWrapper(env)
    policy = load(f"mshab_checkpoints/rl/<task>/pick/{obj_id}/policy.pt")
    rollout(env, policy, n_episodes=200)
    env.close()
```

Output lands at `$MS_ASSET_DIR/robot_success_states/fetch/pick/<obj_id>.pkl`
with `{obj_id, robot_qpos: [Nx15], obj_pose_wrt_base: [Nx7]}`.

**Step 3 -- Mine the affordance set.** The miner runs offline FK on the saved
qpos to recover TCP-in-base, computes
`anchor_obj = inv(obj_pose_wrt_base) * tcp_in_base`, also expresses the TCP
approach axis in the object frame, then clusters anchors with K-means
(default `K=4`). Each component stores the cluster-median anchor, the
cluster-median approach direction, and the cluster-median width:

```bash
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$MS_ASSET_DIR/robot_success_states" \
    --robot fetch \
    --subtask pick \
    --out teemo_sim_probe/configs/affordances.json \
    --n-components 4
```

Design decisions, deliberately:

- **Pick only.** Place success requires `~is_grasped` with TCP at
  `ee_rest_world_pose`, so `inv(obj) * tcp` from place data would encode the
  robot's rest pose rather than an object affordance.
- **FK is required.** If SAPIEN / the Fetch URDF is unavailable the miner
  aborts rather than writing `[0, 0, 0]` placeholder anchors -- placeholders
  would produce valid-looking but false runtime relations.
- **Canonical keys.** Both miner and runtime call
  `canonical_affordance_key`, which strips `env-N_` prefixes and `-N`
  instance suffixes (`env-0_024_bowl-3 -> 024_bowl`).
- **Coverage warnings.** The miner warns if any of the 10 known YCB objects
  has no `.pkl` in `pick/`; the runtime simply emits no affordance edges for
  missing entries (graceful degradation, no crash).
- **Source = success-rest, not mid-grasp.** `FetchCollectRobotInitWrapper`
  fires on `info["success"]`, which for pick happens after the robot has
  returned to `ee_rest_world_pose` while still holding the object. The grasp
  is rigid so the object moves with the TCP and `inv(obj) * tcp` still
  encodes the grasp anchor and approach axis. Mid-trajectory frames with
  `is_grasped=True` (read from the filtered IL `.h5` trajectories) are a
  sensible upgrade if the current relations look noisy.

**Step 4 -- Sanity-check the output.** Open
`teemo_sim_probe/configs/affordances.json` and eyeball each entry: e.g.
`024_bowl` anchors should sit near the rim z; `003_cracker_box` anchors on
the long faces; widths should fall in roughly `[0.02, 0.06] m` for Fetch.
After this the runtime starts emitting `orientation-alignment` and
`gripper-width-alignment` (plus anchor-based planar-distance / height-offset)
for any visible / persistent interactive target whose canonical key is in the
asset.

## obs_mode vs. the released policies

The MS-HAB release includes PPO and SAC checkpoints. The probe builds the env
in `rgb+depth+segmentation`; the policy receives the wrapped depth/state
observation, while the probe reads `segmentation` and RGB directly from
`env.unwrapped` for graph extraction. SAC Fetch checkpoints also use MS-HAB's
`FetchActionWrapper` by default, matching the stationary-head action masking
used during training.

## Usage

ManiSkill (fully runnable, no checkpoint):

```bash
python -m teemo_sim_probe.run_ms_probe \
    --env-id PickCube-v1 \
    --steps 40 \
    --actions random \
    --video
```

Longer ManiSkill probe with periodic saves:

```bash
python -m teemo_sim_probe.run_ms_probe \
    --env-id PickCube-v1 \
    --steps 100 \
    --actions random \
    --save-every 10 \
    --video
```

MS-HAB with a released checkpoint:

```bash
# download checkpoints first:
#   huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/tidy_house/pick/all \
    --steps 60 \
    --video
```

Object-specific SAC checkpoints: the runner infers the matching task plan
from the checkpoint path. This uses
`task_plans/set_table/pick/train/024_bowl.json`:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 \
    --save-every 20 \
    --width 128 \
    --height 128 \
    --video
```

MS-HAB merges each subtask target under an internal name such as `obj_0`. By
default the probe resolves this handle to the actual per-environment actor
name such as `env-0_024_bowl-3` so the active target and visible
segmentation object share one graph node. To preserve the internal MS-HAB
name instead, add `--mshab-object-name merged`; the default is `actual`. In
either mode the graph includes exactly one of the two names, never both.

If `policy.pt` is missing the MS-HAB runner falls back to random actions so
the graph pipeline can still be exercised.

## Milestone order

1. PickCube-v1, num_envs=1 -- print seg map (excl. id 0), build ee + object
   nodes
2. masks + node names overlay
3. planar-distance, height-offset from tcp / object pose
4. contact from pairwise forces; grasp from `is_grasping`
5. temporal labels (K=5)
6. MS-HAB PickSubtaskTrain-v0 + active-target persistence
7. open / close handle nodes
8. affordance relations (object-frame anchors + approach directions, pick-mined)
9. ManiSkill (non-HAB) checkpoints + BEHAVIOR-1K adapter

## Layout

```
teemo_sim_probe/
  run_ms_probe.py          run_mshab_probe.py
  adapters/  privileged_state.py  policy_loader.py
  core/      schema.py  mask_extractor.py  node_builder.py
             affordance.py  relation_rules.py  temporal_buffer.py
             persistence.py  graph_builder.py
  viz/       overlay.py  graph_draw.py  video_writer.py
  configs/   thresholds.yaml  loader.py  affordances.json
  tools/     build_affordances.py
  tests/     test_relation_rules.py  test_affordance.py
```
