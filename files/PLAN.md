# TEEMO eligibility-based relation vocabulary — implementation plan

## What changes and why
Replace the flat `ee--object` relation set (emitted for every object) with an
eligibility split done in two stages:
1. **Classify** each object node as `static_object` or `interactive_object`
   (node_builder.classify_pair_types), stored on `node.attributes["pair_type"]`.
2. **Emit only the eligible relations** per class (relation_rules):
   - static_object  : planar-distance, height-offset, contact  (CENTER-based)
   - interactive_object : planar-distance, height-offset, orientation-alignment,
                          contact, grasp  (+ gripper-width-alignment)  (ANCHOR-based)
   - object--object : contact, support  (unchanged)
Temporal *-change / *-transition are produced automatically by temporal_buffer.

## Key decisions
- **Eligibility = "has an affordance set"** (the doc's own criterion), with name
  fallbacks for not-yet-mined objects and a forced-interactive rule for the
  MS-HAB active handle link. Avoids a brittle hardcoded furniture list.
- **Frames**: distances stay in WORLD frame (consistent with the rest of the
  pipeline + tuned thresholds). orientation-alignment is a relative ANGLE
  between two world-frame approach axes, so the world/robot-frame distinction
  washes out — no new transform path needed.
- **Anchor-based interactive spatial**: planar-distance / height-offset for
  interactive objects are measured from the selected affordance anchor (argmin
  TCP→anchor), falling back to object center when no asset exists. This folds
  the old `tcp-affordance-alignment` distance into the doc's planar/height names.
- **Affordance components: K=4 per object** (good YCB default). Each component
  now also stores `approach_dir_obj_frame` (object-frame unit vector, cluster
  median) so orientation-alignment has a frame to align against.
- **TCP approach axis** is a config constant (`grasp.tcp_approach_axis_local`,
  default +Z) so it's tunable per robot.

## Files
- core/affordance.py          (FULL REPLACEMENT) — adds approach_dir, transform_approach_dir, has_affordance; schema_version 2 (back-compat with v1 assets)
- core/relation_rules.py      (FULL REPLACEMENT) — eligibility split + orientation-alignment
- diffs/node_builder_*.diff   — hints + classify_pair_types()
- diffs/graph_builder.diff    — call classify_pair_types before edge build
- diffs/temporal_buffer*.diff — register orientation-alignment-change; anchor-bound reset
- diffs/thresholds.diff       — orientation_alignment(+change) bins; tcp axis
- diffs/build_affordances.diff— mine approach_dir; schema_version 2

## Apply order
1. Drop in the two FULL files.
2. Apply node_builder, graph_builder, temporal_buffer, thresholds, build_affordances diffs.
3. Re-run the miner with the new approach-dir support to regenerate affordances.json
   (old v1 assets still load; they just won't emit orientation-alignment).
4. Update tests/test_relation_rules.py expectations (static vs interactive sets)
   and the README relation list.
