# DreamerV3 + ManiSkill-HAB + BEHAVIOR-1K

## Setup

### ManiSkill-HAB

**1. Clone this repo and its dependencies**

```bash
git clone https://github.com/chickbong221/ReLDreamer.git
cd ReLDreamer
git clone https://github.com/haosulab/ManiSkill.git
git clone https://github.com/arth-shukla/mshab.git
```

**2. Install dependencies**

```bash
bash install_maniskill.sh
```

**3. Download NVIDIA userspace drivers**

```bash
mkdir -p $HOME/nvidia-userspace
cd $HOME/nvidia-userspace
wget https://us.download.nvidia.com/tesla/570.133.20/NVIDIA-Linux-x86_64-570.133.20.run
```

**4. Create data folder and home alias**

```bash
mkdir -p /mnt/data/$USER
ln -sfn /mnt/data/$USER $HOME/mnt_data
```

**5. Download simulation assets**

```bash
export MS_ASSET_DIR=/mnt/data/$USER
mkdir -p $MS_ASSET_DIR/output

for dataset in ycb ReplicaCAD ReplicaCADRearrange; do
    python -m mani_skill.utils.download_asset "$dataset"
done
```

**6. Submit training job**

```bash
sbatch run_ms.sh
```

---

### BEHAVIOR-1K (OmniGibson)

> **Requires a separate conda env.** Isaac Sim conflicts with the `dreamer` env — `install_behavior1k.sh` creates a dedicated `behavior` env automatically.

**1. Install dependencies**

```bash
bash install_behavior1k.sh
# To also download BEHAVIOR-1K assets (~50-200 GB) in one shot:
bash install_behavior1k.sh --dataset --accept-nvidia-eula --accept-dataset-tos
```

**2. Set asset and data paths** — same conventions as ManiSkill above:

```bash
export OMNIGIBSON_ASSET_PATH=/mnt/data/$USER/og_assets
export OMNIGIBSON_HEADLESS=1   # required on headless servers
```

Add both to `~/.bashrc` to persist. `logdir` is configured in `dreamerv3/configs.yaml`, same as ManiSkill.

**3. Train**

```bash
conda activate behavior
python dreamerv3/main.py \
  --configs behavior1k \
  --task behavior1k_picking_up_trash
```

Task names map to activity names in `BEHAVIOR-1K/bddl3/bddl/activity_definitions/` (e.g. `washing_dishes`, `cleaning_floors`).

To override the robot/scene/sensor config, edit or copy `embodied/envs/behavior1k_cfg/default.yaml` and pass `--env.behavior1k.config_path <path>`.
