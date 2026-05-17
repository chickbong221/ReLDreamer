# DreamerV3 + ManiSkill (GPU Parallel Rendering)

`ManiSkill/` and `mshab/` are **not tracked** by this repository (listed in
`.gitignore`). They must be cloned separately — `install.sh` does this for
you automatically.

`tdmpc2/` contains modifications to ManiSkill's stock TD-MPC2 baseline.
`install.sh` copies it into `ManiSkill/examples/baselines/tdmpc2/` and then
removes the root-level folder to keep the working tree clean.

## Installation

Run the provided script from the repository root:

```bash
bash install.sh
```

**Vulkan** must be installed separately on your system. On a headless server:

```bash
sudo apt-get install -y libvulkan1 vulkan-tools
vulkaninfo --summary   # verify
```

Full Vulkan setup guide:
https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan

---

## Running DreamerV3 on ManiSkill tasks (RGB)

### Command format

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs maniskill_rgb \
  --task maniskill_<ENV_ID> \
  --logger.wandb_project <your-project> \
  --logger.wandb_entity  <your-entity> \
  --logger.wandb_name    dreamerv3-<ENV_ID>-rgb-42
```

Replace `<ENV_ID>` with any task ID from the tables below,
e.g. `--task maniskill_PickCube-v1`.

To continue a stopped run, reuse the same `--logdir`.

### What `maniskill_rgb` sets

| Config key | Value | Meaning |
|---|---|---|
| `env.maniskill.obs_mode` | `rgb` | RGB pixel observations from onboard camera sensors |
| `env.maniskill.image_size` | `128` | Camera resolution — 128 × 128 pixels per sensor |
| `env.maniskill.num_envs` | `32` | Parallel GPU environments batched in one call |
| `env.maniskill.sim_backend` | `gpu` | PhysX GPU-accelerated simulation |
| `env.maniskill.control_mode` | `pd_joint_delta_pos` | Joint-space delta position control |
| `env.maniskill.num_frames` | `1` | No frame stacking — 3-channel RGB to CNN encoder |
| `batch_size` | `8` | Small batch due to large `[N, 128, 128, 3]` tensors |
| `replay.size` | `100000` | Compact replay buffer (pixel storage is expensive) |
| `run.train_ratio` | `128` | Gradient steps per environment step |
| `run.steps` | `4e6` | Total environment steps |
| `jax.prealloc` | `false` | JAX does not preallocate VRAM — required to share GPU with ManiSkill |

Any key can be overridden from the command line, e.g.:

```bash
--env.maniskill.num_envs 16 --env.maniskill.image_size 64
```

## Available ManiSkill tasks

All tasks below work with `--task maniskill_<ENV_ID>`.

### Tabletop manipulation (Panda robot)

| ENV_ID | Description |
|---|---|
| `PickCube-v1` | Pick a red cube and lift it to a target height |
| `PushCube-v1` | Push a cube to a goal position on the table |
| `StackCube-v1` | Stack a red cube on top of a green cube |
| `StackPyramid-v1` | Stack three cubes into a pyramid |
| `PokeCube-v1` | Poke a cube through a target hole |
| `PullCube-v1` | Pull a cube towards the robot |
| `PullCubeTool-v1` | Use a T-shaped tool to pull a cube |
| `PlaceSphere-v1` | Place a sphere into a bowl |
| `RollBall-v1` | Roll a ball to a goal position |
| `PushT-v1` | Push a T-shaped block to a target pose |
| `LiftPegUpright-v1` | Reorient a peg to stand upright |
| `PegInsertionSide-v1` | Insert a peg into a box from the side |
| `PlugCharger-v1` | Insert a charger plug into a socket |
| `TwoRobotPickCube-v1` | Two Pandas cooperatively pick a cube |
| `TwoRobotStackCube-v1` | Two Pandas cooperatively stack cubes |
| `PickSingleYCB-v1` | Pick one of many YCB household objects *(needs YCB assets — see below)* |

> `PickSingleYCB-v1` requires the YCB asset pack:
> ```bash
> python -m mani_skill.utils.download_asset ycb
> ```

### Mobile manipulation (Unitree G1 / H1 humanoid)

| ENV_ID | Description |
|---|---|
| `UnitreeG1TransportBox-v1` | G1 humanoid transports a box across the scene |
| `UnitreeG1PlaceAppleInBowl-v1` | G1 humanoid places an apple in a bowl |
| `UnitreeG1Stand-v1` | G1 humanoid balance / stand task |
| `UnitreeH1Stand-v1` | H1 humanoid balance / stand task |
| `OpenCabinetDoor-v1` | Mobile manipulator opens a cabinet door |

### Locomotion / classical control

| ENV_ID | Description |
|---|---|
| `MS-CartpoleBalance-v1` | Classic cartpole balance |
| `MS-CartpoleSwingUp-v1` | Cartpole swing-up from hanging |
| `MS-AntWalk-v1` | Ant quadruped walks forward |
| `MS-AntRun-v1` | Ant quadruped runs forward |
| `MS-HopperStand-v1` | Hopper stands upright |
| `MS-HopperHop-v1` | Hopper hops forward |
| `MS-HumanoidStand-v1` | Humanoid stands upright |
| `MS-HumanoidWalk-v1` | Humanoid walks forward |
| `MS-HumanoidRun-v1` | Humanoid runs forward |
| `MS-HumanoidStandHard-v1` | Humanoid stand (harder variant) |
| `MS-HumanoidWalkHard-v1` | Humanoid walk (harder variant) |
| `MS-HumanoidRunHard-v1` | Humanoid run (harder variant) |

### Quadruped

| ENV_ID | Description |
|---|---|
| `AnymalC-Reach-v1` | AnymalC reaches a target position |
| `AnymalC-Spin-v1` | AnymalC spins in place |
| `UnitreeGo2-Reach-v1` | Unitree Go2 reaches a target position |

### Dexterity

| ENV_ID | Description |
|---|---|
| `RotateSingleObjectInHandLevel0-v1` | In-hand object rotation — easiest |
| `RotateSingleObjectInHandLevel1-v1` | In-hand object rotation — harder |
| `RotateValveLevel0-v1` | Rotate a valve — level 0 |
| `RotateValveLevel1-v1` | Rotate a valve — level 1 |
| `RotateValveLevel2-v1` | Rotate a valve — level 2 |
| `RotateValveLevel3-v1` | Rotate a valve — level 3 |
| `RotateValveLevel4-v1` | Rotate a valve — level 4 |

---

## Running DreamerV3 on ManiSkill-HAB tasks

### Setup

**1. Install MS-HAB** (if not already done):

```bash
git clone https://github.com/haosulab/ManiSkill.git -b mshab --single-branch
pip install -e ManiSkill
pip install -e mshab   # from this repo's mshab/ subdir, or your own clone
```

**2. Download simulation assets** (ReplicaCAD scenes + rearrangement data, a few GB):

```bash
# Default install path: ~/.maniskill/data
# To change: export MS_ASSET_DIR=/your/path
for dataset in ycb ReplicaCAD ReplicaCADRearrange; do
    python -m mani_skill.utils.download_asset "$dataset"
done
```

This is everything needed for RL. Task plans and spawn data are bundled inside
the `ReplicaCADRearrange` download at
`$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/`.

> The large HuggingFace dataset (~490 GB, `arth-shukla/MS-HAB-*`) contains
> demonstration trajectories for behaviour cloning and diffusion policy.
> **It is not needed for RL training.**

### Available environments

| ENV_ID | Description | Episode steps | Rewards |
|---|---|---|---|
| `PickSubtaskTrain-v0` | Fetch robot picks a target YCB object | 200 | Dense + normalised |
| `PlaceSubtaskTrain-v0` | Fetch robot places held object at goal | 200 | Dense + normalised |
| `OpenSubtaskTrain-v0` | Fetch robot opens a drawer or cabinet door | 200 | Dense + normalised |
| `CloseSubtaskTrain-v0` | Fetch robot closes a drawer or cabinet door | 200 | Dense + normalised |
| `NavigateSubtaskTrain-v0` | Fetch robot navigates to a goal pose | 200 | Dense + normalised |
| `SequentialTask-v0` | Full long-horizon task (Pick → Place → …) — **evaluation only** | — | None |

All subtask environments use the **Fetch** mobile manipulator with
`pd_joint_delta_pos` control.

Available long-horizon task contexts (`mshab_task`):

| Value | Description |
|---|---|
| `tidy_house` | Pick and place objects to tidy a house |
| `prepare_groceries` | Organise groceries in a kitchen |
| `set_table` | Set a table with plates and utensils |

### Training command

Pass `--env.maniskill.mshab_task` to activate MS-HAB mode. The wrapper
automatically loads the matching task plans and spawn data from the downloaded
assets — no manual file editing required.

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs maniskill_rgb \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task tidy_house \
  --env.maniskill.mshab_split train \
  --logger.wandb_project <your-project> \
  --logger.wandb_entity  <your-entity> \
  --logger.wandb_name    dreamerv3-mshab-pick-rgb-42
```

Replace `PickSubtaskTrain-v0` with any training ENV_ID above, and
`mshab_task` with any of `tidy_house`, `prepare_groceries`, or `set_table`.

**`mshab_split`** is `train` by default; set to `val` to use the validation
scenes (21 scenes vs 63 for train).

**`num_envs` recommendation:** the train split has 63 scenes and the val split
has 21. For balanced scene coverage use a multiple of 63 (e.g. 63, 126, 189)
for train or a multiple of 21 for val. Any other value still works — the
wrapper disables the scene-balance assertion automatically — but some scenes
will be sampled more than others.

## Sequential Training and Transfer Evaluation

This feature trains a DreamerV3 agent sequentially on a set of tasks, then
fine-tunes on a new task starting from the pretrained checkpoint. The goal is
to test whether pretraining on Phase 1 tasks accelerates learning on the new
task and to measure how much the agent forgets Phase 1 tasks after Phase 2.

### Pattern

```
Phase 1:  TaskA → ckpt_A → TaskB (from ckpt_A) → ckpt_B → ... → ckpt_N
Phase 2a: TaskNew (from ckpt_N)       ← pretrained branch
Phase 2b: TaskNew (from scratch)      ← baseline branch (optional)
Eval:     pretrained checkpoint evaluated on {TaskA … TaskN, TaskNew}
```

Phases 2a and 2b both log eval metrics during training via the existing
`evalfn` in `train.py`. Overlapping their W&B runs under the same group shows
the transfer advantage directly on the x-axis of training steps.

### Compatibility requirement

All Phase 1 tasks **and** the Phase 2 task must share the **same obs/act
space** — same robot, same `control_mode`, same `obs_mode`. If state
dimensions differ between tasks, `eval_multitask.py` will skip incompatible
tasks with a warning.

Recommended task groups:
- **ManiSkill-HAB subtasks** — all use the Fetch robot + `pd_joint_delta_pos`
- **Panda tabletop tasks** — tasks that happen to share the same state dim

### Files (all inside `sequential/`)

```
sequential/
├── train_sequential.py   # orchestrates all phases, calls eval_multitask.py
├── eval_multitask.py     # loads a checkpoint, evals on each task, prints table
└── config.yaml           # config file for the pipeline
```

No existing files are modified. All sequential training logs are written under
`~/logdir/sequential/` (configurable), kept separate from normal DreamerV3
training logs which go to `~/logdir/dreamer/`.

### Setup

Edit `sequential/config.yaml` to define your tasks and steps:

```yaml
logdir_base: ~/logdir/sequential/run1   # separate from ~/logdir/dreamer/

dreamer_configs:
  - maniskill_rgb

phase1_tasks:
  - PickCube-v1
  - PushCube-v1
  - StackCube-v1

phase2_task: PlugCharger-v1

steps_per_phase1_task: 2_000_000
phase2_steps:          2_000_000

run_baseline: false    # set true or use --run_baseline flag

eval_episodes: 5
eval_envs: 4

extra_flags:           # passed verbatim to dreamerv3/main.py for every run
  - --logger.wandb_project
  - my-project
  - --logger.wandb_group
  - sequential_run1
```

### Run (from repo root)

```bash
# Sequential training without baseline
python sequential/train_sequential.py --config sequential/config.yaml

# With baseline (also trains Phase 2 from scratch for comparison)
python sequential/train_sequential.py --config sequential/config.yaml --run_baseline
```

`train_sequential.py` launches each DreamerV3 training run as a **separate
subprocess**. This gives each phase a clean JAX/CUDA/ManiSkill state and
allows the pipeline to be interrupted and resumed — re-running the script
will skip phases whose logdir already exists (DreamerV3 auto-resumes from
checkpoint if the logdir is present).

### Logdir layout

```
~/logdir/sequential/run1/          ← separate from ~/logdir/dreamer/
├── phase1/
│   ├── PickCube-v1/    ← full DreamerV3 logdir (metrics.jsonl, ckpt/, config.yaml)
│   ├── PushCube-v1/    ← loaded PickCube-v1 checkpoint, trained on PushCube-v1
│   └── StackCube-v1/   ← loaded PushCube-v1 checkpoint — final Phase 1 checkpoint
├── phase2/
│   ├── pretrained/     ← loaded Phase 1 final ckpt, trained on PlugCharger-v1
│   └── baseline/       ← trained on PlugCharger-v1 from scratch (if --run_baseline)
└── eval/
    ├── pretrained/
    │   └── results.json   ← per-task metrics for pretrained checkpoint
    └── baseline/
        └── results.json   ← per-task metrics for baseline checkpoint
```

### Standalone evaluation (from repo root)

You can run `eval_multitask.py` independently on any checkpoint:

```bash
python sequential/eval_multitask.py \
  --checkpoint ~/logdir/sequential/run1/phase2/pretrained/ckpt \
  --train_task PlugCharger-v1 \
  --tasks PickCube-v1 PushCube-v1 StackCube-v1 PlugCharger-v1 \
  --configs maniskill_rgb \
  --episodes 5 \
  --eval_envs 4 \
  --logdir ~/logdir/sequential/run1/eval/pretrained \
  --wandb_project my-project
```

`--train_task` must be the task the checkpoint was last trained on — it sets
the obs/act space the agent expects. `--tasks` is the list to evaluate; any
task whose obs/act space does not match is skipped with a warning.

The script prints a summary table and saves `results.json`:

```
----------------------------------------------------------------
Task                            success_once        return
----------------------------------------------------------------
PickCube-v1                     0.82                0.63
PushCube-v1                     0.74                0.57
StackCube-v1                    0.61                0.44
PlugCharger-v1                  0.55                0.39
----------------------------------------------------------------
```
