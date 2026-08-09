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
`scenegraph` — vertices are the end effector plus whitelisted object instances,
facts are `(i, rho, j)` triples carrying an absolute state and a temporal
change. See [scenegraph/README.md](../scenegraph/README.md) for the graph
contract.

`teemo_sim_probe/` is the frozen predecessor: a self-contained offline demo of
the single-camera, CNN-pooled, target-conditioned version, kept runnable for
comparison and for drawing figures. It carries its own copy of every module it
needs, so nothing in the training path imports it and it never has to be kept
in step.

The graph posterior runs on every replay timestep during training. DINO runs
only in the collector, so the appearance vectors in replay are fixed for the
whole run. Imagination touches neither: the semantic prior supplies `ĝ_t`,
which conditions the low-level dynamics.

## Prerequisites

The runtime fails loud at construction if the mined assets are missing. All of
them live in `scenegraph/configs/` and are mined by `scenegraph/tools/`:

* `affordances.json`
* `subtask_whitelists/<subtask>_<target>.json` — one per `(subtask, target)`
  your task plans reach, for **both** the train and eval splits
* `instructions.npz` — the frozen language embedding every method reads

One command mines all three. Start with `--dry-run`: it prints the coverage
report and every subcommand without running any of them, and collection costs
sim-hours.

```bash
python -m scenegraph.tools.prepare_assets \
  --mshab-task set_table prepare_groceries tidy_house \
  --subtask pick --clean --dry-run
```

Drop `--dry-run` to execute. `--clean` removes the previous artifacts first;
without it a whitelist for an object no longer in your task plans survives the
rebuild, because the miners only write the keys they mined.

Whitelist coverage is verified at startup for `pick` and `place`; a missing
file raises before the first rollout.

`graph.whitelist_union` picks which whitelist binds. A per-target file answers
"what may exist while picking this object"; the merged `<subtask>_all.json`
answers "what may exist in these scenes at all". `auto` uses the merged file
when `mshab_obj` is `all` and the per-target one otherwise.

The merged file also switches off the same-instance filter in
`apply_whitelist`. It has to: every object is `interacted` in its own file, so
the union marks all of them that way, and the filter would leave only the
active target — collapsing the union back to one object.

## Train

```bash
python -m dreamerv3.main \
  --configs maniskill_rgb mshab \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_obj 024_bowl \
  --env.maniskill.num_envs 189 \
  --run.steps 10e6 \
  --logdir $HOME/logdir/mshab/$(date +%Y%m%d_%H%M%S)/pick-024_bowl-graph \
  --logger.wandb_name dreamerv3-graph-set_table-pick-024_bowl
```

`mshab` must come **last**: it overrides `maniskill_rgb`'s `obs_mode` with
`rgb+segmentation`, drops `image_size` to 112, and sets `graph.enabled: true`.
Reversed, you get 128px `rgb` with the graph still enabled and no segmentation
to build it from. Stacking on `maniskill_rgb` is what makes this run one knob
away from the baseline arm — same `control_mode`, `train_ratio`, and logger.

Both presets set `jax.prealloc: false`, and it is not optional: SAPIEN renders
through Vulkan on the same device, and a preallocating JAX starves it into
`CUDA_ERROR_ILLEGAL_ADDRESS` partway through training. DINO runs on that device
too, in the collector only.

To run the same preset **without** the semantic state, as the baseline arm:

```bash
  --env.maniskill.obs_mode depth \
  --env.maniskill.graph.enabled False
```

With no graph keys in the observation space the agent drops the semantic path
entirely: no posterior, no decoder, no graph losses.

## Observation contract

Two cameras, stored separately and never fused. Camera index 0 is
`fetch_head`, 1 is `fetch_hand`.

```
image_head         [112,112,3]   uint8
image_hand         [112,112,3]   uint8
state              [D]           float32
instruction        [768]         float32
graph_node_ent     [24]          uint16
graph_node_app     [24,2,384]    float16
graph_node_bbox    [24,2,4]      float16
graph_node_target  [24]          uint8
graph_edge_*       [1024]        uint8
```

Nothing derivable is stored. The model reads validity, per-camera visibility
and per-camera appearance support back off the content:

```
valid            = graph_node_ent != 0
camera_visible   = bbox[...,1] > bbox[...,0] and bbox[...,3] > bbox[...,2]
appearance_known = abs(graph_node_app).sum(-1) > 0
edge_valid       = graph_edge_rel != 0
temp_mask        = graph_edge_temp != 0
```

**Visibility** is immediate and has no threshold: one segmentation pixel makes
a node visible in that camera, and exclusive box maxima guarantee a one-pixel
node still has positive extent. A node visible in *either* camera is globally
visible; a node visible in neither stays a valid vertex for the rest of the
episode (`k_persist: -1`).

**Appearance** is frozen DINOv2 (`dinov2_vits14_reg`, 8x8 patches at 112px),
pooled per camera under that camera's fractional patch coverage and cached per
`(env, node, camera)`. A camera that loses sight of a node keeps its last
embedding; a camera that has never seen it stays at exactly zero. The box does
not persist — an unseen camera packs a zero box every frame.

**Target** marks the vertex the current subtask acts on, which changes per
subtask under `mshab_obj: all`. It is all-zero when the active object cannot be
resolved to a vertex; `log/graph_target_missing` reports how often, since a
dark flag is otherwise indistinguishable from a resolved one.

## Configuration

**`env.maniskill.graph`** — what the environment emits.

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | emit graph observations at all |
| `whitelist_dir` | `''` | mined whitelists; falls back to `thresholds.yaml` |
| `profile` | `room_scale` | threshold profile for bin fallbacks |
| `cameras` | `[fetch_head, fetch_hand]` | camera order for the stored axis; the first also renders overlays |
| `n_max` | `24` | vertex capacity including the ee node; must stay under 256. The merged whitelist admits 19 categories plus the ee, so the surplus is headroom for duplicate instances — overflow drops the newcomer, which can be the target |
| `e_max` | `1024` | fact capacity per frame; overflow drops spatial before affordance before physical |
| `k_persist` | `-1` | negative keeps a registered vertex for the whole episode |
| `dino_model` | `dinov2_vits14_reg` | registers keep artifact tokens out of the patch features |
| `dino_res` | `112` | must be a multiple of the patch size 14 |
| `dino_weights` | `''` | local checkpoint; empty pulls from `torch.hub` |
| `app_dim` | `384` | stored feature width, checked against the loaded model at startup |
| `staleness_enabled` | `true` | retain object–object physical state for nodes that left the view |

**`agent.graph`** — the semantic posterior.

| key | default | meaning |
|---|---|---|
| `layers` | `2` | message-passing rounds |
| `units` | `256` | node / fact width — **the size preset's `.*\.units` wildcard overrides this** |
| `embed` | `64` | embedding-table width |
| `app_dim` | `384` | stored DINO width; must match `env.maniskill.graph.app_dim` |
| `app` | `64` | learned projection width *per camera* |
| `bbox` | `8` | learned box projection width per camera |
| `reverse_edges` | `True` | add the reversed fact with a direction flag, so information reaches the ee node |
| `condition_on_deter` | `True` | condition node and fact encodings on `h_t`, per the method. This puts the GNN inside the scan — set `False` to hoist it out and trade fidelity for speed |
| `entity_vocab` | `64` | placeholder, overwritten from the mined whitelists at startup; do not set by hand |

A vertex is `[AppProj_c(a_c), BBoxProj_c(b_c) for each camera, EntityEmbed(id),
TargetEmbed(flag)]` = 2*64 + 2*8 + 64 + 64 = 272, projected to `units`. Each
per-camera block is gated to zero when that camera has no embedding or no box,
so a projection bias cannot stand in for missing data. There is no global
visibility scalar: the per-camera boxes already carry it.

The target is a two-token embedding rather than a widened entity vocabulary. A
category means the same thing whether or not it is this episode's goal, so
splitting `actor:bowl` into goal and non-goal ids would halve the data behind
each and buy nothing.

**`agent.dyn.rssm`** — `semstoch: 16`, `semclasses: 16`, `semlayers: 1`,
`free_nats: 1.0`.

**`agent.loss_scales`** — `node`, `relabs`, `reltemp`, `semtgt`, `semdyn`,
`semrep`. `node` is the mean of the appearance, box and visibility heads.
`semtgt` reads goal identity out of the semantic state; it is small because a
handful of bits saturates it.

Every width other than the four above resolves from the selected size preset.
Nothing in the model hard-codes a channel count or token width, so `size1m`
through `size400m` all build.

## What to watch

| metric | reads as |
|---|---|
| `train/loss/semdyn`, `semrep` | semantic KL, **after** the free-nats floor |
| `train/semdyn_raw`, `semrep_raw` | the same KL before clipping. Below 1.0 means the semantic KL contributes no gradient; report it rather than changing `free_nats` mid-run |
| `train/relabs_acc`, `reltemp_acc` | relation-state reconstruction. Saturating immediately means the heads are trivial; narrow the node width |
| `train/node_app_var` | spread of the appearance target across vertices. Read this before `node_app_cos` |
| `train/node_bbox_iou` | box reconstruction, current camera-visible entries only |
| `train/node_vis_acc` | per-camera visibility. The hand camera sees few objects, so a constant-zero head scores well here — read it against the positive rate |
| `train/semtgt_acc` | goal identity recovered from the semantic state alone. Low means the semantic KL is evicting the target and imagination is running blind to it |
| `train/semtgt_frac` | fraction of steps carrying a target label. Read it before `semtgt_acc` — an accuracy over almost no steps means nothing |
| `train/sem_ent` | semantic posterior entropy |
| `episode/log/graph_overflow_drops` | vertices the registry could not seat. Nonzero means retained nodes from an earlier subtask are displacing current ones |
| `episode/log/graph_fact_drops` | facts truncated at `e_max`. Nonzero means spatial edges are being lost — `graph_pack` keeps physical, then affordance, then spatial |
| `episode/log/graph_target_missing` | frames whose graph flags no target vertex. Active-object resolution fails open for a whole episode at a time, so anything but near-zero means part of the run is training ungrounded |
| `episode/log/graph_cache_entries` | entries held by the caches that outlive an episode. A sawtooth under the cap is healthy; a climb that never levels off is a leak, and `GraphObsBuilder.cache_stats()` names which container |
| `replay/ram_gb` | roughly 93 KiB/step, images and appearance dominating |

## Instruction input

`instruction` is a frozen T5 embedding of the active subtask, looked up per env
per step from `subtask_pointer` and the un-merged task plans, so a run where
each episode picks a different object gets a different vector each episode. It
is built offline by `scenegraph/tools/build_instruction_embeddings.py` and
loaded through `env.maniskill.instruction_table`.

Every method in a comparison loads the same table. The encoder conditions on
the vector and the policy reads it back out of the latent; the decoder does not
reconstruct it, since it is constant within an episode.

`mshab_holdout_objs` — comma-separated canonical keys, e.g.
`--env.maniskill.mshab_holdout_objs 024_bowl,013_apple` — splits object
categories for the few-shot protocol: training excludes them, the finetune env
(`mshab_holdout_mode: only`) and eval see only them. A frozen language encoder
is what makes the held-out rows meaningful: they are already positioned
relative to the categories training saw, which a learned embedding table
cannot do.

## Ablations

All config-only, no code changes:

```bash
--agent.loss_scales.relabs 0 --agent.loss_scales.reltemp 0   # no relation supervision
--agent.loss_scales.semtgt 0                                 # target flag as input only, unsupervised
--agent.graph.reverse_edges False                            # one-directional message passing
--agent.graph.condition_on_deter False                       # h_t-free encoder, GNN outside the scan
--env.maniskill.graph.e_max 128                              # tighter fact budget
--env.maniskill.instruction_table random.npz                 # control: same keys, no language features
```

## Known limitations

Deliberate for this version, listed so they are not rediscovered as bugs:

* **Cross-subtask retention.** `merge_persistent` runs after `apply_whitelist`,
  so a vertex admitted under one MS-HAB subtask stays after the whitelist
  rebinds and keeps its old entity key. With `k_persist: -1` and an append-only
  registry it can consume slots out of `n_max` and displace a current-subtask
  entity. Not fixed; watch `graph_overflow_drops`.
* **Terminal graphs are one frame stale.** The vector env auto-resets inside
  `step`, so the terminal transition re-emits the previous packed graph. Every
  graph loss and both semantic KLs are masked at `is_last`; the image, state,
  reward and low-level KL are not, because those *are* the true final values.
  The stale graph still reaches the semantic posterior at that step.
* **The graph decoder is not a semantic bottleneck.** It reconstructs from the
  posterior node representations, which were built from the same appearance and
  boxes, with the entity embedding present. It grounds the posterior trunk; it
  does not force information through sampled `g_t`.
* **Appearance at 112px is 8x8 patches.** Each patch covers 1/64 of the frame,
  so a small object's embedding carries surrounding context. Empirical.
* **Two live DINO models.** Training and evaluation each own one; a third is
  built and discarded during observation-space discovery.
* **Replay size.** ~93 KiB per transition, ~65 GB at `replay.size: 700000`.

## Verify first

Neither the world model nor the graph encoder has been exercised on real
rollouts. Run the suite before launching a long job:

```bash
python -m unittest discover scenegraph/tests
```

That covers the graph contract, the frozen encoder's batching and pooling
(`test_dino.py`, with the checkpoint mocked), the appearance cache and terminal
guard (`test_appearance_cache.py`), and — on a machine with jax — agent
construction across size presets, RSSM losses, decoder masks and terminal
masking (`test_world_model.py`) plus pooling invariance and every derived mask
(`test_semantic_posterior.py`). The jax and torch suites **skip silently**
without those libraries, so confirm the run reports no skips in the training
env before trusting them.

Then the live path, which is the only thing that touches sensors:

```bash
python -m scenegraph.tools.smoke_graph_obs --num-envs 4 --steps 120
```

It checks named-camera shapes and dtypes, state parity with upstream
flattening, that both cameras pool a nonzero appearance, that terminal frames
carry the true final image rather than the post-reset render, and prints
measured bytes per transition.
