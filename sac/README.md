# SAC on MS-HAB `set_table` / `pick` / `024_bowl`

Vectorized SAC training on the bowl-pick subtask of `set_table`. Mirrors
`mshab/configs/sac_pick.yml`: depth input from both `fetch_head` (stationary)
and `fetch_hand` (moves with gripper), 189 parallel envs, batch 512, 1M
replay. No frame stacking. Two variants:

- **plain**: depth only.
- **graph**: depth + oracle scene graph (needs teemo assets mined first).

## Pipeline

```
[graph variant only]
  1. mine teemo assets (affordances + subtask whitelists)

[always]
  2. sac.main --configs pick_bowl [--graph.enabled True]
```

## Run — plain depth

```
python -m sac.main \
    --configs pick_bowl \
    --task maniskill_PickSubtaskTrain-v0
```

## Run — depth + oracle graph

Mine the teemo assets once (bowl target on set_table):

```
export MS_ASSET_DIR=/root/.maniskill
STATES_DIR="$MS_ASSET_DIR/data/robot_success_states"

python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root mshab_checkpoints/rl \
    --task set_table --subtask pick --obj 024_bowl \
    --n-success 30 --num-envs 8 --no-skip-done

python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$STATES_DIR" \
    --robot fetch --subtask pick \
    --out teemo_sim_probe/configs/affordances.json

python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$STATES_DIR" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists
```

Then train:

```
python -m sac.main \
    --configs pick_bowl \
    --task maniskill_PickSubtaskTrain-v0 \
    --graph.enabled True
```

## Common overrides

```
--env.maniskill.num_envs 32        # fewer parallel envs if 189 is too heavy
--agent.buffer_size 300_000        # smaller replay
--run.total_steps 10_000_000       # shorter run
--graph.camera fetch_hand          # switch graph seg to the hand camera
```

Full CLI dotted keys map onto `configs.yaml`.
