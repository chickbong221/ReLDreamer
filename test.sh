  python -m sac.main --configs pick_bowl --task maniskill_PickSubtaskTrain-v0 \
      --graph.enabled False \
      --agent.buffer_size 40000 \
      --run.total_steps 200000 --run.log_every 2000 \
      --logger.wandb_name 'leak-A-baseline' \
      --logdir 'logdir/leak_A_{timestamp}'

  # B — full graph: obs_mode = "depth+segmentation" + all teemo compute (known-leaks)
  python -m sac.main --configs pick_bowl --task maniskill_PickSubtaskTrain-v0 \
      --graph.enabled True \
      --agent.buffer_size 40000 \
      --run.total_steps 200000 --run.log_every 2000 \
      --logger.wandb_name 'leak-B-full-graph' \
      --logdir 'logdir/leak_B_{timestamp}'

  # C — graph enabled but bypass_teemo: still renders + reads seg, no teemo compute, no pair-force queries
  python -m sac.main --configs pick_bowl --task maniskill_PickSubtaskTrain-v0 \
      --graph.enabled True --graph.bypass_teemo True \
      --agent.buffer_size 40000 \
      --run.total_steps 200000 --run.log_every 2000 \
      --logger.wandb_name 'leak-C-bypass-teemo' \
      --logdir 'logdir/leak_C_{timestamp}'