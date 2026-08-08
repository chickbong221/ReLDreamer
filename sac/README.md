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

Mine the assets once. They land in `scenegraph/configs/` and are shared with
the DreamerV3 pipeline:

```
export MS_ASSET_DIR=/root/.maniskill

python -m scenegraph.tools.prepare_assets \
    --mshab-task set_table --subtask pick --clean
```

Add `--dry-run` first to see the coverage report and the subcommands without
spending the sim time.

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
