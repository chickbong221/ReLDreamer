# SAC (with optional oracle scene graph)

Vectorized SAC on ManiSkill 3 / MS-HAB. Supports `state`, `rgb`, `depth`, and
`rgb + oracle scene graph`.

## Pipeline

```
1. mine affordances + whitelists (teemo_sim_probe/tools, once per asset bump)
2. sac.main --configs <preset> --task <env>
```

Graph obs requires the whitelists to be mined first. State/rgb/depth
presets run without any teemo assets.

## Run

State-only:
```
python -m sac.main --configs maniskill_state --task maniskill_PickCube-v1
```

RGB:
```
python -m sac.main --configs maniskill_rgb --task maniskill_PickCube-v1
```

MS-HAB depth:
```
python -m sac.main --configs mshab --task maniskill_PickSubtaskTrain-v0 \
    --env.maniskill.mshab_task pick
```

RGB + oracle scene graph (needs step 1):
```
python -m sac.main --configs mshab_graph --task maniskill_PickSubtaskTrain-v0 \
    --env.maniskill.mshab_task pick
```

## Mine graph assets (only for the graph preset)

```
export MS_ASSET_DIR=/root/.maniskill
STATES_DIR="$MS_ASSET_DIR/data/robot_success_states"

for SUB in pick open close; do
    python -m teemo_sim_probe.tools.collect_robot_success_states \
        --ckpt-root mshab_checkpoints/rl \
        --task set_table --subtask "$SUB" \
        --n-success 30 --num-envs 8 --no-skip-done
done

python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "$STATES_DIR" \
    --robot fetch --subtask pick \
    --out teemo_sim_probe/configs/affordances.json

python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "$STATES_DIR" \
    --out-dir teemo_sim_probe/configs/subtask_whitelists
```

See `teemo_sim_probe/README.md` for details on the mining pipeline.

## Overrides

Dotted CLI overrides on any yaml key:
```
python -m sac.main --configs mshab_graph --task maniskill_PickSubtaskTrain-v0 \
    --graph.camera fetch_head --graph.e_max 384 --agent.batch_size 256
```
