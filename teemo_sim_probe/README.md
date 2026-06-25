# TEEMO simulation probe

The probe builds a fixed-size manipulation graph from simulator segmentation
and physics. MS-HAB is read-only: all integration and collection behavior lives
under `teemo_sim_probe/adapters/`.

## Graph contract

There are two node types:

* `ee` — the end effector;
* `object` — every non-robot actor or articulation link, including handles,
  drawers, counters, and free objects.

A handle has no special admission rule. It appears only when the robot touched
it during a successful offline rollout and it is present in the active
whitelist. Supporters and handles use separate stable identities.

## Offline pipeline

`FetchCollectContactDataWrapper` buffers each vector environment independently.
A buffer is committed only when that environment succeeds; failed episodes are
discarded. One committed rollout contains:

1. the task target key, used only to name/select the offline asset;
2. every scene entity contacted by any robot link;
3. direct supporters of contacted entities.

Support stops after one hop. There is no contact BFS and no recursive supporter
closure. The task target is not injected as a member just because it is active.
Robot links provide interaction evidence but are never members.

The collector writes schema-v3 pickles:

```text
{
  _schema_version: 3,
  obj_id,
  entity_key,
  subtask_type,
  robot_qpos,
  obj_pose_wrt_base,
  interaction_rollouts
}
```

The legacy pose arrays remain for affordance mining. Affordances use the same
entity-key namespace for actors and articulation links. Incidental contact is
enough for whitelist membership but not enough to create an affordance.

`build_subtask_whitelists` takes the union across successful rollouts:

```text
members = robot-interacted entities
        ∪ direct supporters of those entities
```

Counts are stored for audit and never filter membership.

## Runtime pipeline

The graph builder advances every environment step:

```text
active (subtask, target key)
  -> load per-subtask whitelist
  -> build all current non-robot segmentation candidates
  -> merge ordinary short-term persistence
  -> hard whitelist gate
  -> classify by affordance + whitelist role
  -> role-aware capacity (interacted/task, support, other)
  -> stable slots
  -> absolute and temporal relations
```

There is no invisible active-target injection and no local-contact admission
path. The active target identifier selects an asset; it does not create a node.

Name-based scene filters are deliberately absent. Relevance comes from the
whitelist, so a visible whitelisted drawer or counter keeps its segmentation
mask and appears in the overlay.

Previously visible nodes may remain for `k_persist` frames. Their last observed
relations are marked `stale`, carry `observed_frame` and `age`, render as blue
dashed edges, and do not update temporal histories.

## Setup

Set the ManiSkill root (the parent of `data/`):

```bash
export MS_ASSET_DIR=/root/.maniskill
```

Collect schema-v3 successful rollouts:

```bash
python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root mshab_checkpoints/rl \
    --n-success 30 --num-envs 8 \
    --no-skip-done
```

`--no-skip-done` is required once when replacing schema-v2 assets.

Mine affordances:

```bash
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --robot fetch --subtask pick \
    --out teemo_sim_probe/configs/affordances.json \
    --n-components 4
```

If released/local `open` or `close` checkpoints are available, collect them
with `--subtask open` or `--subtask close`, then add their qualified link
affordances without replacing the actor entries:

```bash
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --robot fetch --subtask open \
    --out teemo_sim_probe/configs/affordances.json \
    --n-components 4 --merge-existing
```

Mine whitelists:

```bash
python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$MS_ASSET_DIR/data/robot_success_states" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists
```

Run an MS-HAB probe:

```bash
python -m teemo_sim_probe.run_mshab_probe \
    --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
    --steps 200 --save-every 20 --video
```

`--save-every` controls rendering only. Selection and temporal state still
advance every simulator step.

## Tests

```bash
python -m unittest discover teemo_sim_probe/tests
```
