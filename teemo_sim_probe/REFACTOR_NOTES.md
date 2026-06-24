# Track C + Track A + Track B Refactor — Change Log

Captured for offline review since this machine has no MS-HAB / ManiSkill /
SAPIEN libs and cannot run the probe. Take this file with you when you next
sit at the GPU machine.

Treat each section as a separate landing block. For runnable commands, see
`README.md`.

---

## 1. Files modified

### `teemo_sim_probe/core/mask_extractor.py`
- `MaskAccumulator.__init__(seg=None)` now optionally retains the seg image
  and exposes `visible_seg_ids` as the set of non-zero seg ids present in
  this frame.
- **Reason:** Bug 2 needs to know which seg ids are actually visible right
  now to filter the synthetic local-contact admission path. Stashing it on
  the accumulator avoids changing `build_nodes`' return signature.

### `teemo_sim_probe/core/node_builder.py`
- `build_nodes` constructs `MaskAccumulator(H, W, seg=seg)` (passes the seg
  image through).
- **Reason:** Plumbing for the above.

### `teemo_sim_probe/core/selector.py`
- **Deleted:** `score()`, `topk_with_refresh()`, oracle, weights, `tau_age`,
  `tau_dist`, `n_refresh`. The soft rule-score machinery is gone (Track A).
- **New `apply_whitelist(nodes)`** — hard eligibility gate. Drops every
  non-`ee` node whose `match_key` is not in the bound whitelist. Raises if
  no whitelist is bound (fail-loud).
- **New `overflow_truncate(nodes)`** — keeps at most `n_slots` nodes, sorted
  by `(planar_distance_to_ee, node_id)`. Distance is the only ordering
  signal and is only consulted here.
- **New `set_whitelist(wl)` / `whitelist` property** — binds the per-subtask
  whitelist at episode reset.
- **New `evict_expired(frame)`** — age-based eviction; the only place
  history entries are dropped (Bug P).
- **`commit(selected_ids, nodes, frame)`** — now snapshots **all visible
  object nodes**, not only selected ones (Bug P). This is the core change
  that lets persistence survive a brief unselection.
- **`expand_local_contact`** — accepts `masks: MaskAccumulator` kwarg. Keys
  synthesized nodes under `canonical_object_key(ent)` instead of
  `local:<name>` (Bug 1). Requires the seg id to be in
  `masks.visible_seg_ids` AND a non-empty mask to be extractable (Bug 2).
  Registers the mask on the accumulator so the overlay sees it. If the
  canonical key already exists, flips `is_local_contact=True` on the
  existing node and merges `segmentation_ids` instead of inserting a
  duplicate.

### `teemo_sim_probe/core/graph_builder.py`
- New helpers `_dedup_by_live_entity`, `_pick_winner`, `_merge_into` —
  collapse nodes that resolve to the same SAPIEN entity after both the
  normal seg-id path and the local-contact path have run (Bug 1,
  defense-in-depth).
- `__init__`: no longer takes `e_domain`; reads `whitelist_dir` from cfg;
  tracks `_whitelist_key` for cache invalidation.
- New `_resolve_and_bind_whitelist(state)` — loads
  `<subtask>_<canonical_target>.json` once per (subtask, target). Fail-loud
  on missing file.
- `step()` pipeline:
  - Calls `_resolve_and_bind_whitelist` before anything else.
  - Removes the old `E_domain.contains()` gate.
  - After `expand_local_contact` runs the dedup pass.
  - Replaces `score + topk_with_refresh` with
    `apply_whitelist + overflow_truncate`.
  - Replaces the per-frame `selector.evict(unselected)` sweep with
    `selector.evict_expired(frame)` (Bug P). Temporal history is purged only
    for truly-expired entries.

### `teemo_sim_probe/core/relation_rules.py`
- `object_object_edges` filter tightened: requires
  `n.valid_mask AND n.segmentation_ids` (Bug 3a). Maskless persistent /
  phantom nodes no longer participate in object–object physics.

### `teemo_sim_probe/core/temporal_buffer.py`
- `update()` no longer seeds `support=False` for every ordered object pair
  (the Cartesian product was the root of Bug 3b). It now seeds **only pairs
  that actually emitted a `contact` or `support` edge this frame**. The
  existing "absence" fallback still maintains correct False-history for
  previously tracked pairs whose endpoints are still in the graph.

### `teemo_sim_probe/configs/thresholds.yaml`
- **Removed** the `e_domain:` block.
- **Removed** `selection.weights`, `n_refresh`, `tau_age`,
  `tau_dist_tabletop`, `tau_dist_room_scale`, `oracle_force_active_target`.
- **Added** `whitelists.dir` (default `subtask_whitelists`).
- **Added** `selection.enable_local_contact` (moved from the deleted
  `e_domain` block).
- `selection` now contains only `n_slots`, `k_persist`,
  `enable_local_contact`.

### `teemo_sim_probe/configs/loader.py`
- No longer imports `load_e_domain`. No longer loads / validates the
  E_domain asset. No longer emits `cfg["e_domain"]` / `cfg["e_domain_set"]`.
- Adds `cfg["whitelists"]` (paths) and `cfg["whitelist_dir"]` (resolved
  absolute path).
- `require_assets=True` now requires only the affordance asset.

### `teemo_sim_probe/tools/collect_robot_success_states.py`
- Swapped the wrapper import to the new
  `FetchCollectContactDataWrapper` (aliased to the same local name to
  minimise diff).
- **Drop-in:** the new wrapper writes to the same output path the old one
  used (`robot_success_states/<robot>/<subtask>/<obj>.pkl`) with a strict
  superset schema, so `build_affordances.py` keeps working unchanged.

### `teemo_sim_probe/tools/setup_assets.sh`
- Step 4 swapped from `build_e_domain` to `build_subtask_whitelists`.
- Path constants updated.

### `teemo_sim_probe/run_ms_probe.py`, `teemo_sim_probe/run_mshab_probe.py`
- Dropped the dead CLI flags: `--n-refresh`, `--oracle-active-target`,
  `--dist-only`.
- Added `--whitelist-dir` override.
- `_apply_ablation_overrides` no longer touches `cfg["e_domain"]`, weights,
  or oracle.

### `teemo_sim_probe/tests/test_relation_rules.py`
- `_node()` now assigns a unique `segmentation_ids=[seg_id]` to every test
  node so they pass the new `object_object_edges` filter (Bug 3a).

### `teemo_sim_probe/tests/test_selector.py`
- Wholly rewritten. `ScoreTests`, `RefreshTests`, `OracleTests` deleted
  (their behavior no longer exists). New tests:
  - `WhitelistGateTests.test_drops_off_whitelist_actor`
  - `WhitelistGateTests.test_link_specificity` — drawer1 dropped, drawer3
    kept (the OQ1 finding).
  - `WhitelistGateTests.test_fails_loud_without_whitelist`
  - `OverflowTruncationTests.test_keeps_nearest_to_ee`
  - `OverflowTruncationTests.test_tiebreak_by_node_id`
  - `PersistenceTests.test_unselected_visible_node_still_persists` — Bug P
    regression lock.
  - `PersistenceTests.test_evict_expired_drops_only_aged_out` — new method
    coverage.
  - `LocalContactTests.test_adds_ee_touching_entity_under_canonical_key` —
    Bug 1 regression lock.

---

## 2. Files added

### `teemo_sim_probe/core/whitelist.py`
- `match_key(node)` — actor → `canonical_affordance_key`; link → bare name;
  else → bare name.
- `Whitelist` dataclass with `contains(key)` / `roles(key)` / `empty`.
- `load_whitelist(path)` — strict loader; raises `FileNotFoundError` /
  `ValueError` on malformed input.
- `resolve_whitelist_path(dir, subtask, target)` —
  `<subtask>_<target>.json` lookup helper.

### `teemo_sim_probe/adapters/collect_contact_data.py`
- `FetchCollectContactDataWrapper` — extends the upstream collect-data flow
  to ALSO record, at each success frame and for each successful env, the
  pairwise contact graph in a neighborhood of the target (default 1.5 m
  planar radius). Pkl shape `schema_version=2`; preserves `robot_qpos` and
  `obj_pose_wrt_base` so the existing affordance miner still works against
  the same files.
- Writes to `<ASSET_DIR>/robot_success_states/<robot_uid>/<subtask>/<target>.pkl`
  by default — the same path the old wrapper used. Override with
  `out_root=`.
- `mshab/` is untouched; this is a sibling adapter that gets composed into
  the env stack the same way the old one did.

### `teemo_sim_probe/tools/build_subtask_whitelists.py`
- Walks the success-state pkls, BFS-closes from the target over the contact
  pair graph (default `max_hops=2`), normalizes keys via the same
  `match_key` rule the runtime uses, and writes `<subtask>_<target>.json`
  per (subtask, target).
- Knobs: `--max-hops`, `--eps-dz`, `--min-support-frac` (default 0.3,
  lenient), `--min-contact-frac` (default 0.6, strict).

---

## 3. Files deleted

- `teemo_sim_probe/core/e_domain.py`
- `teemo_sim_probe/tools/build_e_domain.py`
- `teemo_sim_probe/adapters/collect_data.py` — replaced by
  `collect_contact_data.py`.

Both E_domain pieces were retired by Track A. `relation_rules.py` never used
`EDomain`. The historic `e_domain.json` asset on disk is no longer read —
leave it on the GPU machine if you want; nothing touches it.

---

## 4. Verification checklist

After re-collect + re-mine on the GPU machine, pick one episode of
`pick_024_bowl` and one of `open_<some_drawer>`. Compare new
`outputs/graph_*.json` against:

- [ ] **No duplicate nodes** — no two object nodes resolve to the same
  physical entity in any frame. (Bug 1)
- [ ] **No maskless graph nodes** — every non-ee object node has
  `pixel_area > 0` AND non-empty `segmentation_ids`. (Bug 2)
- [ ] **No phantom support** — no `support` / `support-transition` edge
  exists between a pair where one endpoint has empty `segmentation_ids`
  (shouldn't happen since they're filtered, but assert anyway). (Bug 3)
- [ ] **Persistence horizon** — manually occlude the target with a long arm
  sweep; verify it remains in the graph for at least `k_persist` (= 5)
  frames after disappearing from the segmentation. (Bug P)
- [ ] **Whitelist gate** — no node whose `match_key` is not in the active
  whitelist appears in `graph.nodes` (excluding `ee` and `<pad:...>`).
  (Track A)
- [ ] **Overflow** — construct a scene with more than `n_slots`=10 eligible
  nodes; assert exactly the 10 nearest to ee (by `pose_world[:2]`) survive,
  tie-broken by `node_id`. (Track A)
- [ ] **Link specificity** — for an `open` subtask whose whitelist names
  `drawer3`, assert sibling drawers (`drawer1`, `drawer2`, …) of the same
  cabinet are NOT in any frame's graph. (Track A3 — the OQ1 invariant)

---

## 5. Known limitations / TODOs flagged during the refactor

- **OQ1 caveat:** link names are bare strings like `drawer3`. If two
  cabinets in the same scene ever share a link name (e.g., two cabinets
  each with their own `drawer3`), the match key will collide and the
  whitelist will admit both. Workable for now; if it bites, prefix the link
  key with the articulation name when the miner emits it and when
  `match_key` resolves it.
- **`build_e_domain.py`'s `_SUPPORT_FIELDS` bug** is now moot since the
  file is deleted, but if you ever resurrect support mining from the JSON
  task plans, note that two of those fields (`goal_rectangle_corners`,
  `goal_pos`) are geometry, not entity names — the old miner could never
  have emitted supports from them.
- **Local-contact one-hop** still runs even after the whitelist gate
  replaces E_domain, since it operates on V_{t-1} not on E_domain. That's
  intentional: a one-hop scratchpad entity that briefly touches the target
  during manipulation gets re-canonicalized to its canonical key; if it's
  in the whitelist the gate keeps it, otherwise the gate drops it. So
  nothing leaks.
