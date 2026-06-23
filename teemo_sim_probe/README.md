# teemo_sim_probe

A standalone probe that extracts TEEMO **manipulation semantic graphs** from
ManiSkill 3 and ManiSkill-HAB (MS-HAB) episodes, and renders them as
mask overlays + node-link diagrams (like masked-based SGG figures).

It lives at the repo root, separate from `dreamerv3/` and `embodied/` — it does
not touch Dreamer.

## What it does

Per frame it produces:

1. `graph_XXXX.json` — nodes + absolute/temporal edges (raw values kept).
2. `overlay_XXXX.png` — per-node segmentation masks + labels on the RGB frame.
3. `graph_XXXX.png` — the semantic graph as a node-link diagram.
4. optional `probe.mp4` — overlay | graph side by side.

Two node types only, matching the draft: `ee` and `object`.

## Key correction baked in: there is no `get_privileged_state()`

Neither ManiSkill 3 nor MS-HAB ships a single "semantic graph state" function.
`adapters/privileged_state.py` **implements that name** by gathering primitives
that *do* exist (verified against the actual source):

| Need            | Real API used                                                       |
|-----------------|---------------------------------------------------------------------|
| seg → entity    | `env.unwrapped.segmentation_id_map` (id 0 = background, excluded)    |
| ee pose         | `agent.tcp_pose` (Fetch, computed) / `agent.tcp.pose` (Panda)        |
| gripper width   | `agent.robot.qpos[-2:].sum()` (Fetch, qpos-sum convention)           |
| fingers         | `agent.finger1_link`, `agent.finger2_link`                          |
| contact         | `scene.get_pairwise_contact_forces(a, b)`                           |
| grasp           | `agent.is_grasping(obj, max_angle=30)` (MS-HAB convention)          |
| MS-HAB targets  | `subtask_objs`, `subtask_articulations`, `task_plan`, `subtask_pointer` |
| Original obj_id | `env.build_config_idx_to_task_plans[bci][tpi].subtasks[ptr].obj_id` |

MS-HAB caveats handled: `subtask_objs[i]` / `subtask_articulations[i]` can be
`None` (close / navigate subtasks); handle links come from
`subtask_articulations[i].links[subtask.articulation_handle_link_idx]`; goal
actors live in `_hidden_objects` and are excluded by default. `task_plan[ptr]`
holds the **merged** name (`obj_0`); the original `024_bowl-3` is recovered
through `build_config_idx_to_task_plans` so affordance lookups work.

## Node detection rules (core/node_builder.py)

1. exclude segmentation id 0 (background)
2. merge gripper / tcp / finger links into the single `ee` node
3. every non-bg, non-robot Actor → `object`
4. every non-robot Link → `object` (articulation parts, handles)
5. exclude helper/goal actors unless `--include-goals`
6. minimum visible-area threshold (`min_pixels` / `min_area_ratio`)
7. add MS-HAB active target even if occluded → `persistent`, empty mask

## Relations (core/relation_rules.py, core/temporal_buffer.py)

- `ee-object`: planar-distance, height-offset (5-bin), contact, grasp
- `ee-target` (MS-HAB only, requires affordance asset):
  - `tcp-affordance-alignment` — distance from TCP to the nearest anchor on
    the active manipulation object, in world frame.
  - `gripper-width-alignment` — signed error
    `current_qpos_sum − preferred_qpos_sum` for the chosen anchor.
  Both edges share one selected component (`a_star`) per object per frame.
  Anchor storage is in **object frame** so the same asset works regardless
  of where the object sits — see [Affordance assets](#affordance-assets).
- `object-object`: mutually exclusive contact or directed support. Support is
  emitted as `supporter → supported` for vertical load-bearing contact; other
  touching remains contact.
- temporal (horizon K=5): signed-change bins for continuous relations;
  gain / lose / maintain / maintain-no for binary predicates.
  `maintain-no-*` is an internal background class and is not exported as a
  semantic graph edge. Affordance histories are reset whenever `a_star`
  switches (preferred-width changes) or the affordance edge disappears, so
  signed-change labels never compare across components.
- `orientation-alignment` and `containment` are intentionally deferred until
  the basic demo validates. TCP orientation is not consumed by any active
  relation (the `ee` node's `pose_world` still carries `[xyz, qw, qx, qy, qz]`
  for future use).

Thresholds live in `configs/thresholds.yaml`, with separate `tabletop`
(ManiSkill) and `room_scale` (MS-HAB) profiles.

## Affordance assets

Affordance relations only fire when there is an entry for the current active
MS-HAB target in `configs/affordances.json`. The repo ships an **empty**
asset — without mined data the runtime emits zero affordance edges (no
crashes, no placeholders).

### What is in the asset

Per canonical YCB object id (10 known objects in MS-HAB:
`002_master_chef_can`, `003_cracker_box`, `004_sugar_box`,
`005_tomato_soup_can`, `007_tuna_fish_can`, `008_pudding_box`,
`009_gelatin_box`, `010_potted_meat_can`, `013_apple`, `024_bowl`):

```json
"024_bowl": {
  "raw_obj_id": "024_bowl",
  "n_samples": 1234,
  "components": [
    { "anchor": [0.000, 0.000, 0.020], "width": 0.045, "n_support": 412 },
    { "anchor": [0.030, 0.000, 0.010], "width": 0.050, "n_support": 397 },
    ...
  ]
}
```

Each component is a pair `(anchor_obj_frame, preferred_width)`:

- **`anchor` is in OBJECT frame** (metres). This is pose-invariant — it's
  "the spot on the bowl where the gripper grips" — so the same value is
  reused frame after frame regardless of where the bowl currently sits.
- **`width` is the gripper qpos-sum** at the success moment: literally
  `robot_qpos[-2] + robot_qpos[-1]` for Fetch. The runtime uses the same
  convention (`adapters/privileged_state.py:compute_gripper_width`) so they
  match exactly. **Do not** mix in the URDF finger-origin offset of
  `+0.03085 m` — it would systematically shift every reading.

At runtime each frame we transform every anchor through the **current**
`obj_pose_world`, pick `a_star` = nearest anchor to the TCP, then emit the
two edges.

### How to collect an affordance set

The miner reads MS-HAB pick-success rollouts saved by
`mshab.envs.wrappers.collect_data.FetchCollectRobotInitWrapper` and emits one
`affordances.json`.

**About the two MS-HAB data artifacts (don't confuse them):**

- `rearrange-dataset/<task>/<subtask>/<obj_id>.h5` — the *IL training
  dataset*, produced by MS-HAB's "Efficient, Controlled Data Generation at
  Scale" pipeline (`scripts/gen_dataset.sh` → `mshab.utils.gen.gen_data`).
  Full trajectories, rule-based event-labeled and filtered. **Not what the
  affordance miner consumes.**
- `robot_success_states/<robot>/<subtask>/<obj_id>.pkl` — the *robot init /
  success-state data*, produced by `FetchCollectRobotInitWrapper`. Just
  `(robot_qpos, obj_pose_wrt_base)` at success moments, one object per file.
  Used internally to build `spawn_data.pt`. **This is what the miner reads.**

**Step 1 — Download MS-HAB's released checkpoints.** Training from scratch
is not required; the released RL checkpoints already cover all 10 YCB
objects. From the existing Usage section:

```bash
huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints
```

Checkpoints live at `mshab_checkpoints/rl/<task>/pick/<obj_id>/policy.pt`.

**Step 2 — Obtain `robot_success_states/`**. Check `$MS_ASSET_DIR/` first —
this directory may already ship as part of the MS-HAB asset bundle (it
backs `spawn_data.pt`). If it is present, skip to Step 3.

If it is missing, collect it yourself with `FetchCollectRobotInitWrapper`:
the wrapper requires single-object task plans (it asserts `len(tp)==1` and
all plans share the same `obj_id`), so you run one rollout per YCB id with
its matching checkpoint:

```python
# sketch: collect_success_states.py
import gymnasium as gym
from mani_skill import ASSET_DIR
from mshab.envs.make import make_env                # MS-HAB env factory
from mshab.envs.wrappers.collect_data import FetchCollectRobotInitWrapper

for obj_id in ["002_master_chef_can", "003_cracker_box", "004_sugar_box",
               "005_tomato_soup_can", "007_tuna_fish_can", "008_pudding_box",
               "009_gelatin_box", "010_potted_meat_can", "013_apple",
               "024_bowl"]:
    env = make_env(...,
        task="tidy_house",   # or set_table for 013_apple/024_bowl variants
        subtask="pick",
        split="train",
        task_plan_fp=f".../task_plans/<task>/pick/train/{obj_id}.json",
        spawn_data_fp=f".../spawn_data/<task>/pick/train/spawn_data.pt",
    )
    env = FetchCollectRobotInitWrapper(env)
    policy = load(f"mshab_checkpoints/rl/<task>/pick/{obj_id}/policy.pt")
    rollout(env, policy, n_episodes=200)            # successes are auto-saved
    env.close()                                     # writes <obj_id>.pkl
```

Output lands at
`$MS_ASSET_DIR/robot_success_states/fetch/pick/<obj_id>.pkl`, each pkl
holding `{obj_id, robot_qpos: [N×15], obj_pose_wrt_base: [N×7]}`.

**Step 3 — Mine the affordance set.** The miner runs offline FK on the saved
qpos to recover TCP-in-base, computes
`anchor_obj = inv(obj_pose_wrt_base) * tcp_in_base`, clusters anchors with
K-means (default `K=4`), and writes per-cluster median anchor + median width:

```bash
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$MS_ASSET_DIR/robot_success_states" \
    --robot fetch \
    --subtask pick \
    --out teemo_sim_probe/configs/affordances.json \
    --n-components 4
```

What the miner does and does NOT do, deliberately:

- **Pick only.** Place success requires `~is_grasped` with the TCP at
  `ee_rest_world_pose`, so `inv(obj) * tcp` from place data learns the
  robot's rest pose, not an object affordance.
- **FK is required.** If SAPIEN / the Fetch URDF is unavailable, the miner
  aborts rather than writing `[0,0,0]` placeholder anchors — placeholders
  would produce valid-looking but false runtime relations.
- **Canonical keys.** Both miner and runtime call
  `canonical_affordance_key`, which strips `env-N_` prefixes and `-N`
  instance suffixes (`env-0_024_bowl-3 → 024_bowl`).
- **Coverage warnings.** The miner warns if any of the 10 known YCB objects
  has no `.pkl` in `pick/`; the runtime will simply emit no affordance edges
  for missing entries (graceful degradation, no crash).
- **Source = success-rest, not mid-grasp.** `FetchCollectRobotInitWrapper`
  fires on `info["success"]`, which for pick happens after the robot has
  returned to `ee_rest_world_pose` while still holding the object. The grasp
  is rigid so the object moves with the TCP and `inv(obj) * tcp` still
  encodes the grasp anchor on the object. A cleaner source — mid-trajectory
  frames with `is_grasped=True` — would require reading the filtered IL
  `.h5` trajectories instead; not done here but a sensible upgrade if the
  current relations look noisy.

**Step 4 — Sanity-check the output.** Open
`teemo_sim_probe/configs/affordances.json` and eyeball each entry: e.g.
`024_bowl` anchors should sit near the rim z; `003_cracker_box` anchors on
the long faces; widths should fall in roughly `[0.02, 0.06] m` for Fetch.
After this, the runtime starts emitting `tcp-affordance-alignment` and
`gripper-width-alignment` relations for any visible/persistent MS-HAB target
whose canonical key is in the asset.

## obs_mode vs. the released policies

The MS-HAB release includes PPO and SAC checkpoints. The probe builds the env in
`rgb+depth+segmentation`; the policy receives the wrapped depth/state
observation, while the probe reads `segmentation` and RGB directly from
`env.unwrapped` for graph extraction. SAC Fetch checkpoints also use MS-HAB's
`FetchActionWrapper` by default, matching the stationary-head action masking
used during training.

## Usage

ManiSkill (fully runnable, no checkpoint):

```bash
cd /root/projects/ReLDreamer

python -m teemo_sim_probe.run_ms_probe \
    --env-id PickCube-v1 \
    --steps 40 \
    --actions random \
    --video
```

Longer ManiSkill probe:

```bash
cd /root/projects/ReLDreamer

python -m teemo_sim_probe.run_ms_probe \
    --env-id PickCube-v1 \
    --steps 100 \
    --actions random \
    --save-every 10 \
    --video
```

MS-HAB with a released checkpoint:

```bash
cd /root/projects/ReLDreamer

# download checkpoints first:
#   huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir /root/projects/ReLDreamer/mshab_checkpoints/rl/tidy_house/pick/all \
    --steps 60 \
    --video
```

For object-specific SAC checkpoints, keep the task plan matched to the
checkpoint object. The runner infers this from the checkpoint path, so this uses
`task_plans/set_table/pick/train/024_bowl.json`:

```bash
cd /root/projects/ReLDreamer

python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir /root/projects/ReLDreamer/mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 \
    --save-every 20 \
    --width 128 \
    --height 128 \
    --video
```

MS-HAB merges each subtask target under an internal name such as `obj_0`. By
default, the probe resolves this handle to the actual per-environment actor name
such as `env-0_024_bowl-3`, so the active target and visible segmentation object
share one graph node. To preserve the internal MS-HAB name instead, add:

```bash
--mshab-object-name merged
```

The default can be stated explicitly with `--mshab-object-name actual`.
In either mode, the graph includes exactly one of the two names, never both.

If `policy.pt` is missing the MS-HAB runner falls back to random actions so the
graph pipeline can still be exercised.

## Milestone order

1. PickCube-v1, num_envs=1 — print seg map (excl. id 0), build ee + object nodes
2. masks + node names overlay
3. planar-distance, height-offset from tcp/object pose
4. contact from pairwise forces; grasp from `is_grasping`
5. temporal labels (K=5)
6. MS-HAB PickSubtaskTrain-v0 + active-target persistence
7. open/close handle nodes
8. affordance relations (object-frame anchors, pick-mined)

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
