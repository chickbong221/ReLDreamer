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
| fingers         | `agent.finger1_link`, `agent.finger2_link`                          |
| contact         | `scene.get_pairwise_contact_forces(a, b)`                           |
| grasp           | `agent.is_grasping(obj, max_angle=30)` (MS-HAB convention)          |
| MS-HAB targets  | `subtask_objs`, `subtask_articulations`, `task_plan`, `subtask_pointer` |

MS-HAB caveats handled: `subtask_objs[i]` / `subtask_articulations[i]` can be
`None` (close / navigate subtasks); handle links come from
`subtask_articulations[i].links[subtask.articulation_handle_link_idx]`; goal
actors live in `_hidden_objects` and are excluded by default.

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
- `object-object`: contact, support
- temporal (horizon K=5): signed-change bins for continuous relations;
  gain / lose / maintain / maintain-no for binary predicates.
  `maintain-no-*` is kept in JSON but `masked=True` (never drawn).
- `orientation-alignment` and `containment` are intentionally deferred until
  the basic demo validates.

Thresholds live in `configs/thresholds.yaml`, with separate `tabletop`
(ManiSkill) and `room_scale` (MS-HAB) profiles.

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
python -m teemo_sim_probe.run_ms_probe \
    --env-id PickCube-v1 --steps 40 --actions random --video
```

MS-HAB with a released checkpoint:

```bash
# download checkpoints first:
#   huggingface-cli download arth-shukla/mshab_checkpoints --local-dir mshab_checkpoints
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/tidy_house/pick/all \
    --steps 60 --video
```

For object-specific SAC checkpoints, keep the task plan matched to the
checkpoint object. The runner infers this from the checkpoint path, so this uses
`task_plans/set_table/pick/train/024_bowl.json`:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 20 --width 128 --height 128 --video
```

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

## Layout

```
teemo_sim_probe/
  run_ms_probe.py          run_mshab_probe.py
  adapters/  privileged_state.py  policy_loader.py
  core/      schema.py  mask_extractor.py  node_builder.py
             relation_rules.py  temporal_buffer.py  graph_builder.py
  viz/       overlay.py  graph_draw.py  video_writer.py
  configs/   thresholds.yaml  loader.py
```
