# Semantic-graph DreamerV3

DreamerV3 with a semantic stochastic state `g` between the recurrent state `h`
and the low-level stochastic state `z`:

```
h_t = f(h_{t-1}, g_{t-1}, z_{t-1}, a_{t-1})     sequence model
g_t ~ q(. | h_t, g_{t-1}, G_t)                  semantic posterior  (reads the scene graph)
ĝ_t ~ p(. | h_t, g_{t-1})                       semantic prior      (used in imagination)
z_t ~ q(. | h_t, g_t, o_t)                      low-level posterior
ŷ_t ~ p(. | [g_t, h_t, z_t])                    obs / reward / continue
```

`G_t` is the task-relevant scene graph built from simulator state by
`teemo_sim_probe` — vertices are the end effector plus whitelisted object
instances, facts are `(i, rho, j)` triples carrying an absolute state and a
temporal change. See [teemo_sim_probe/README.md](../teemo_sim_probe/README.md)
for the graph contract.

Imagination never touches a graph: the semantic prior supplies `ĝ_t`, which
conditions the low-level dynamics. The graph encoder and ground-truth graphs
are needed only for the world-model loss.

## Prerequisites

The runtime fails loud at episode start if the mined assets are missing. Both
come from `teemo_sim_probe/tools/`:

* `teemo_sim_probe/configs/affordances.json`
* `teemo_sim_probe/configs/subtask_whitelists/<subtask>_<target>.json` — one per
  `(subtask, target)` your task plans reach, for **both** the train and eval
  splits.

Mine them with steps 1–3 of the sweep in
[teemo_sim_probe/README.md](../teemo_sim_probe/README.md). Whitelist coverage is
verified at startup for `pick` and `place`; a missing file raises before the
first rollout.

## Train

```bash
python -m dreamerv3.main \
  --configs maniskill_rgb mshab \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_obj 024_bowl \
  --env.maniskill.num_envs 189 \
  --env.maniskill.graph.whitelist_dir teemo_sim_probe/configs/subtask_whitelists \
  --run.steps 10e6 \
  --logdir $HOME/logdir/mshab/$(date +%Y%m%d_%H%M%S)/pick-024_bowl-graph \
  --logger.wandb_name dreamerv3-graph-set_table-pick-024_bowl
```

`mshab` must come **last**: it overrides `maniskill_rgb`'s `obs_mode` with
`depth+segmentation` and sets `graph.enabled: true`. Reversed, you get `rgb` with
the graph still enabled and no segmentation to build it from. Stacking on
`maniskill_rgb` is what makes this run one knob away from the baseline arm —
same `control_mode`, `image_size`, `train_ratio`, and logger.

Both presets set `jax.prealloc: false`, and it is not optional: SAPIEN renders
through Vulkan on the same device, and a preallocating JAX starves it into
`CUDA_ERROR_ILLEGAL_ADDRESS` partway through training.

To run the same preset **without** the semantic state, as the baseline arm:

```bash
  --env.maniskill.obs_mode depth \
  --env.maniskill.graph.enabled False
```

With no graph keys in the observation space the agent drops the semantic path
entirely: no posterior, no decoder, no `sem*` losses.

## Configuration

**`env.maniskill.graph`** — what the environment emits.

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | emit graph observations at all |
| `whitelist_dir` | `''` | mined whitelists; falls back to `thresholds.yaml` |
| `profile` | `room_scale` | threshold profile for bin fallbacks |
| `cameras` | `[fetch_head, fetch_hand]` | appearance is one pooled depth per camera, in this order |
| `primary_camera` | `fetch_head` | camera whose masks feed the overlay renderer |
| `n_max` | `11` | vertex capacity including the ee node; must stay under 256 |
| `e_max` | `256` | fact capacity per frame; overflow drops spatial before affordance before physical |
| `staleness_enabled` | `true` | retain object–object physical state for nodes that left the view |

**`agent.graph`** — the semantic posterior.

| key | default | meaning |
|---|---|---|
| `layers` | `2` | message-passing rounds |
| `units` | `256` | node / fact width |
| `embed` | `64` | embedding-table width |
| `reverse_edges` | `True` | add the reversed fact with a direction flag, so information reaches the ee node |
| `condition_on_deter` | `True` | condition node and fact encodings on `h_t`, per the method. This puts the GNN inside the scan — set `False` to hoist it out and trade fidelity for speed |
| `entity_vocab` | `64` | placeholder, overwritten from the mined whitelists at startup; do not set by hand |

**`agent.dyn.rssm`** — `semstoch: 16`, `semclasses: 16`, `semlayers: 1`.

**`agent.loss_scales`** — `semapp` (appearance, low by design: it is one scalar
per camera), `semvis`, `semabs`, `semtemp`, `semdyn`, `semrep`.

## What to watch

| metric | reads as |
|---|---|
| `train/loss/semdyn`, `semrep` | semantic KL. Collapsing to the free-nats floor means `g` carries nothing — the lever is the low-level prior's dependence on `g`, not `beta` |
| `train/semabs_acc`, `semtemp_acc` | relation-state reconstruction. Saturating immediately means the heads are trivial; narrow the node width |
| `train/semvis_acc` | visibility reconstruction |
| `train/sem_ent` | semantic posterior entropy |
| `replay/ram_gb` | graph keys add roughly 2 KB/step on top of the RSSM carry |

## Ablations

All config-only, no code changes:

```bash
--agent.loss_scales.semabs 0 --agent.loss_scales.semtemp 0   # no relation supervision
--agent.graph.reverse_edges False                            # one-directional message passing
--agent.graph.condition_on_deter False                       # h_t-free encoder, GNN outside the scan
--env.maniskill.graph.e_max 128                              # tighter fact budget
```

## Verify first

Neither the world model nor the graph encoder has been exercised on real
rollouts. Run both suites before launching a long job:

```bash
python -m unittest discover teemo_sim_probe/tests
```

That covers the graph contract everywhere, and — on a machine with jax — also
`test_world_model.py` (agent construction against this config, RSSM losses,
decoder heads, graph-free imagination) and `test_semantic_posterior.py`
(permutation and padding invariance of the pooled token). Those two **skip
silently without jax**, so confirm the run reports no skips in the training env
before trusting them.
