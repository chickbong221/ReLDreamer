# SAC on MS-HAB `set_table` / `pick` / `024_bowl`

Vectorized SAC training on the bowl-pick subtask of `set_table`. RGB input,
300K replay. Two variants:

- **plain**: RGB only.
- **graph**: RGB + oracle scene graph (needs teemo assets mined first).

## Pipeline

```
[graph variant only]
  1. mine teemo assets (affordances + subtask whitelists)

[always]
  2. sac.main --configs pick_bowl [--graph.enabled True]
```

## Run — plain RGB

```
python -m sac.main \
    --configs pick_bowl \
    --task maniskill_PickSubtaskTrain-v0
```

## Run — RGB + oracle graph

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
--agent.buffer_size 500_000        # bigger replay
--env.maniskill.num_envs 8         # fewer parallel envs
--run.total_steps 10_000_000       # shorter run
--graph.camera fetch_head          # top-down camera for graph
```

Full CLI dotted keys map onto `configs.yaml`.
