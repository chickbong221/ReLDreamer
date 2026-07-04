"""``get_privileged_state`` -- the adapter the simulator does NOT give you.

There is no built-in ``env.get_privileged_state()`` in ManiSkill 3 or MS-HAB.
This module implements that name by gathering the primitives that *do* exist:

  * ``env.unwrapped.agent``                    (Fetch / Panda)
  * ``env.unwrapped.scene``                    -> get_pairwise_contact_forces(...)
  * ``env.unwrapped.segmentation_id_map``      seg-id int -> Actor / Link
  * ``agent.tcp`` / ``agent.tcp_pose``         end-effector pose
  * ``agent.finger1_link`` / ``finger2_link``  contact + grasp
  * ``agent.is_grasping(obj, max_angle=30)``   grasp predicate (MS-HAB convention)

For MS-HAB it also exposes the task internals used to decide which object must
persist as the active manipulation target:

  * ``env.unwrapped.subtask_objs``             (entries may be None)
  * ``env.unwrapped.subtask_goals``            (entries may be None)
  * ``env.unwrapped.subtask_articulations``    (entries may be None)
  * ``env.unwrapped.task_plan``                list of *Subtask dataclasses
  * ``env.unwrapped.subtask_pointer``          per-env current subtask index

Everything is torch-batched with a leading env dimension; helpers here index a
single ``env_idx`` and return plain python / numpy where convenient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Small typed snapshot returned to the rest of the pipeline.
# --------------------------------------------------------------------------- #
@dataclass
class PrivilegedState:
    env: Any                       # unwrapped env (kept for force queries)
    agent: Any
    scene: Any
    env_idx: int

    is_mshab: bool

    # End-effector.
    ee_links: List[Any] = field(default_factory=list)   # Link objs to merge into ee
    tcp_pose_world: Optional[np.ndarray] = None          # [7] xyz + wxyz

    # Gripper width in qpos-sum convention: qpos[-2] + qpos[-1] for Fetch.
    # Matches the miner (tools/build_affordances.py). None for non-Fetch agents
    # or when qpos is unavailable.
    gripper_width: Optional[float] = None

    # Segmentation id -> Actor/Link (ManiSkill primitive).
    seg_id_map: Dict[int, Any] = field(default_factory=dict)

    # Robot link set (for "is this a robot link?" tests).
    robot_links: set = field(default_factory=set)

    # MS-HAB active manipulation handles for this env_idx (any may be None).
    active_obj: Optional[Any] = None
    # Pre-resolution merged MS-HAB handles (span all envs). Physics queries
    # can use these when the resolved per-env wrapper is not row-aligned.
    active_obj_merged: Optional[Any] = None
    active_handle_link_merged: Optional[Any] = None
    active_articulation: Optional[Any] = None
    active_handle_link: Optional[Any] = None
    active_subtask_type: Optional[str] = None
    # Original (pre-merge) obj_id for the current subtask -- e.g. "024_bowl-3".
    # MS-HAB rewrites task_plan[ptr].obj_id to "obj_<num>" during _merge_*; the
    # original lives in env.build_config_idx_to_task_plans[bci][tpi].subtasks[ptr].
    active_obj_id: Optional[str] = None

    # ----- queries the relation rules call ------------------------------- #
    def pairwise_force_vector(self, a: Any, b: Any) -> np.ndarray:
        """World-frame contact-force vector between two entities for this env.

        ManiSkill builds contact queries from ``zip(a._bodies, b._bodies)``.
        MS-HAB can mix full-span robot links with per-env actors/links, so row
        k is not necessarily env k. Slice to single-env views when needed.
        """
        if a is None or b is None:
            return np.zeros(3, dtype=float)
        ra = _obj_index_for_env(a, self.env_idx)
        rb = _obj_index_for_env(b, self.env_idx)
        if ra is None or rb is None:
            return np.zeros(3, dtype=float)
        if ra != rb:
            a = _slice_view_for_env(a, self.env_idx)
            b = _slice_view_for_env(b, self.env_idx)
            if a is None or b is None:
                return np.zeros(3, dtype=float)
            ra = rb = 0
        forces = _to_np(self.scene.get_pairwise_contact_forces(a, b))
        if forces.ndim == 1:
            return forces.astype(float)
        if ra >= forces.shape[0]:
            return np.zeros(3, dtype=float)
        return np.asarray(forces[ra], dtype=float)

    def pairwise_force(self, a: Any, b: Any) -> float:
        """Scalar contact-force magnitude between two entities for this env."""
        return float(np.linalg.norm(self.pairwise_force_vector(a, b)))

    def ee_object_contact_force(self, obj: Any) -> float:
        """Sum of both finger contact forces against ``obj`` (MS-HAB style)."""
        if obj is None:
            return 0.0
        f1 = self.pairwise_force(self.agent.finger1_link, obj)
        f2 = self.pairwise_force(self.agent.finger2_link, obj)
        return f1 + f2

    def _finger_open_dir(self, finger_link, sign: float) -> Optional[np.ndarray]:
        """World-frame Fetch gripper-opening direction for this env."""
        arr = entity_pose_world_array(finger_link, self.env_idx)
        if arr is None:
            return None
        w, x, y, z = arr[3], arr[4], arr[5], arr[6]
        ydir = np.array(
            [
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ],
            dtype=float,
        )
        return sign * ydir

    def _is_grasping_manual(
        self, obj: Any, max_angle: int, min_force: float = 0.5
    ) -> bool:
        """Per-env reimplementation of Fetch.is_grasping for sliced objects."""
        f1 = getattr(self.agent, "finger1_link", None)
        f2 = getattr(self.agent, "finger2_link", None)
        if f1 is None or f2 is None:
            return False
        lf = self.pairwise_force_vector(f1, obj)
        rf = self.pairwise_force_vector(f2, obj)
        lmag = float(np.linalg.norm(lf))
        rmag = float(np.linalg.norm(rf))
        if lmag < min_force or rmag < min_force:
            return False
        ld = self._finger_open_dir(f1, -1.0)
        rd = self._finger_open_dir(f2, +1.0)
        if ld is None or rd is None:
            return False

        def _angle_deg(d: np.ndarray, f: np.ndarray, fmag: float) -> float:
            denom = float(np.linalg.norm(d)) * fmag + 1e-12
            c = float(np.dot(d, f)) / denom
            return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

        return (
            _angle_deg(ld, lf, lmag) <= max_angle
            and _angle_deg(rd, rf, rmag) <= max_angle
        )

    def is_grasping(self, obj: Any, max_angle: int = 30) -> bool:
        """Env-consistent grasp predicate for full-span and per-env objects."""
        if obj is None:
            return False
        if _obj_index_for_env(obj, self.env_idx) is None:
            return False

        num_envs = getattr(self.scene, "num_envs", None)
        objs = getattr(obj, "_objs", None)
        n_rows = len(objs) if objs is not None else 1

        query = None
        if num_envs is not None and n_rows == num_envs:
            query = obj
        elif self.active_obj_merged is not None:
            try:
                same = _entity_for_env(obj, self.env_idx) is _entity_for_env(
                    self.active_obj_merged, self.env_idx
                )
            except Exception:
                same = False
            if same:
                query = self.active_obj_merged

        if query is None or not hasattr(self.agent, "is_grasping"):
            return self._is_grasping_manual(obj, max_angle)

        try:
            g = self.agent.is_grasping(query, max_angle=max_angle)
        except TypeError:
            g = self.agent.is_grasping(query)
        g = _to_np(g).reshape(-1)
        if self.env_idx >= g.shape[0]:
            return False
        return bool(g[self.env_idx])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _scene_idxs_list(entity) -> Optional[List[int]]:
    try:
        s = entity._scene_idxs
    except AttributeError:
        return None
    return s.tolist() if hasattr(s, "tolist") else list(s)


def _obj_index_for_env(entity, env_idx: int) -> Optional[int]:
    """Row of ``entity._objs`` that lives in parallel env ``env_idx``."""
    if entity is None:
        return None
    s = _scene_idxs_list(entity)
    if s is None:
        objs = getattr(entity, "_objs", None)
        if objs is None:
            return 0
        return env_idx if env_idx < len(objs) else None
    try:
        return s.index(env_idx)
    except ValueError:
        return None


def get_tcp_pose(agent) -> Any:
    """Fetch defines ``tcp_pose`` (computed); Panda exposes ``tcp.pose``."""
    if hasattr(agent, "tcp_pose"):
        return agent.tcp_pose
    return agent.tcp.pose


def compute_gripper_width(agent, env_idx: int) -> Optional[float]:
    """Width = qpos[-2] + qpos[-1] (matches MS-HAB miner / collect_data.py).

    Fetch qpos layout is 15-D: [base 3 | head 2 | torso 1 | arm 7 | gripper 2],
    so qpos[-2:] are the two finger prismatic joints. We deliberately do NOT
    use ``||finger1.pose.p - finger2.pose.p||`` here -- the URDF joint origins
    add a constant +0.03085 m to that distance, which would systematically bias
    every ``gripper-width-alignment`` reading relative to the mined preferred
    widths.
    """
    if agent is None or not hasattr(agent, "robot"):
        return None
    try:
        qpos = _to_np(agent.robot.qpos)
    except Exception:
        return None
    if qpos.ndim == 0:
        return None
    if qpos.ndim == 1:
        row = qpos
    elif qpos.ndim == 2:
        if env_idx < 0 or env_idx >= qpos.shape[0]:
            return None
        row = qpos[env_idx]
    else:
        return None
    if row.shape[0] < 2:
        return None
    w = float(row[-2] + row[-1])
    if not np.isfinite(w):
        return None
    return w


def pose_to_world_array(pose, env_idx: int) -> np.ndarray:
    """SAPIEN Pose -> [x, y, z, qw, qx, qy, qz] for one env."""
    p = _to_np(pose.p)
    q = _to_np(pose.q)
    if p.ndim == 2:
        p = p[env_idx]
    if q.ndim == 2:
        q = q[env_idx]
    return np.concatenate([p, q]).astype(float)


def get_ee_links(agent) -> List[Any]:
    """Links to fold into the single ``ee`` node.

    Fetch / Panda both cache ``tcp``, ``finger1_link``, ``finger2_link`` in
    ``_after_init``. We start with those; the run scripts can print the seg map
    and extend this set (e.g. wrist/hand links) once empirically observed.
    """
    links = []
    for attr in ("tcp", "finger1_link", "finger2_link"):
        link = getattr(agent, attr, None)
        if link is not None:
            links.append(link)
    return links


def _robot_link_set(agent) -> set:
    try:
        return set(agent.robot.get_links())
    except Exception:
        return set()


def _looks_like_mshab(env) -> bool:
    return all(
        hasattr(env, a)
        for a in ("subtask_objs", "task_plan", "subtask_pointer")
    )


def _subtask_type(subtask) -> Optional[str]:
    # *Subtask dataclasses expose ``.type`` ("pick"/"place"/"open"/"close"/...).
    return getattr(subtask, "type", None)


def _entity_for_env(entity, env_idx: int):
    """Return the underlying simulator entity represented at ``env_idx``."""
    if entity is None:
        return None
    objs = getattr(entity, "_objs", None)
    if objs is None:
        return entity

    try:
        scene_idxs = _to_np(entity._scene_idxs).reshape(-1).tolist()
        return objs[scene_idxs.index(env_idx)]
    except (AttributeError, ValueError, IndexError, TypeError):
        pass

    try:
        if len(objs) == 1:
            return objs[0]
        return objs[env_idx]
    except (IndexError, TypeError):
        return entity


def per_env_segmentation_id_map(env, env_idx: int) -> Dict[int, Any]:
    """Segmentation-id -> Actor/Link map valid for one parallel env.

    ManiSkill's global ``env.segmentation_id_map`` keys wrappers by
    ``_objs[0].per_scene_id``. MS-HAB can load heterogeneous actors and
    articulations across vector envs, so the same integer id can refer to
    different entities in different sub-scenes.
    """
    scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
    cache = scene.__dict__.setdefault("_teemo_per_env_seg_maps", {})
    cached = cache.get(env_idx)
    if cached is not None:
        return cached

    res: Dict[int, Any] = {}
    for actor in scene.actors.values():
        if getattr(actor, "merged", False):
            continue
        row = _obj_index_for_env(actor, env_idx)
        objs = getattr(actor, "_objs", None)
        if row is None or not objs or row >= len(objs):
            continue
        res[int(objs[row].per_scene_id)] = actor

    for art in scene.articulations.values():
        if getattr(art, "merged", False):
            continue
        if _obj_index_for_env(art, env_idx) is None:
            continue
        for link in art.links:
            row = _obj_index_for_env(link, env_idx)
            objs = getattr(link, "_objs", None)
            if row is None or not objs or row >= len(objs):
                continue
            res[int(objs[row].entity.per_scene_id)] = link

    cache[env_idx] = res
    return res


def entity_pose_world_array(entity, env_idx: int) -> Optional[np.ndarray]:
    """Pose row of ``entity`` for parallel env ``env_idx``.

    ``pose.p`` rows follow ``entity._objs`` order, not global env order.
    """
    pose = getattr(entity, "pose", None)
    if pose is None:
        return None
    row = _obj_index_for_env(entity, env_idx)
    if row is None:
        return None
    p = _to_np(pose.p)
    q = _to_np(pose.q)
    if p.ndim == 2:
        if row >= p.shape[0]:
            return None
        p = p[row]
    if q.ndim == 2:
        q = q[row] if row < q.shape[0] else q[0]
    return np.concatenate([p, q]).astype(float)


def _slice_view_for_env(entity, env_idx: int):
    """Return a single-row Actor/Link view of ``entity`` scoped to ``env_idx``."""
    objs = getattr(entity, "_objs", None)
    if objs is None:
        return entity
    if len(objs) == 1:
        return entity if _obj_index_for_env(entity, env_idx) is not None else None
    row = _obj_index_for_env(entity, env_idx)
    if row is None:
        return None

    scene = getattr(entity, "scene", None)
    cache = scene.__dict__.setdefault("_teemo_sliced_views", {}) if scene else {}
    key = (id(entity), env_idx)
    hit = cache.get(key)
    if hit is not None and hit[0] is entity:
        return hit[1]

    import torch
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.link import Link

    sidx = torch.tensor([env_idx], dtype=torch.int64)
    if type(entity).__name__ == "Link":
        view = Link.create([objs[row]], scene, sidx)
        view.articulation = getattr(entity, "articulation", None)
    else:
        view = Actor.create_from_entities([objs[row]], scene, sidx)
    try:
        view.name = f"{getattr(entity, 'name', 'entity')}@{id(entity)}@env{env_idx}"
    except Exception:
        pass
    cache[key] = (entity, view)
    return view


def _resolve_actual_entity(entity, seg_id_map: Dict[int, Any], env_idx: int):
    """Map an MS-HAB merged handle to its per-env segmentation wrapper.

    MS-HAB names task-level merged actors ``obj_0``, ``obj_1``, etc. The
    segmentation map contains another ManiSkill wrapper around the same SAPIEN
    entity, but with its actual scene name (for example
    ``env-0_024_bowl-3``). Returning that wrapper keeps pose/contact APIs valid
    and lets the persistent target merge with its visible segmentation node.
    """
    if entity is None:
        return None
    target = _entity_for_env(entity, env_idx)
    target_name = getattr(target, "name", None)

    for candidate in seg_id_map.values():
        candidate_target = _entity_for_env(candidate, env_idx)
        if candidate_target is target:
            return candidate

    # Identity is the reliable path, but matching the concrete SAPIEN name is
    # a useful fallback across wrapper/proxy implementations.
    if target_name is not None:
        for candidate in seg_id_map.values():
            candidate_target = _entity_for_env(candidate, env_idx)
            if getattr(candidate_target, "name", None) == target_name:
                return candidate
    return entity


def _alias_segmentation_entity(
    seg_id_map: Dict[int, Any], alias, env_idx: int
) -> Dict[int, Any]:
    """Use a merged MS-HAB handle for matching segmentation entries.

    This is the inverse of ``_resolve_actual_entity``. It preserves every
    segmentation id and mask while ensuring merged-name mode creates only the
    ``obj_x`` node, rather than both ``obj_x`` and the actual-name node.
    """
    if alias is None:
        return seg_id_map
    target = _entity_for_env(alias, env_idx)
    target_name = getattr(target, "name", None)
    aliased = dict(seg_id_map)

    for seg_id, candidate in seg_id_map.items():
        candidate_target = _entity_for_env(candidate, env_idx)
        same_entity = candidate_target is target
        same_name = (
            target_name is not None
            and getattr(candidate_target, "name", None) == target_name
        )
        if same_entity or same_name:
            aliased[seg_id] = alias
    return aliased


def _active_mshab_handles(
    env,
    env_idx: int,
    seg_id_map: Optional[Dict[int, Any]] = None,
    object_name: str = "actual",
) -> Dict[str, Any]:
    """Resolve the current subtask's object / articulation / handle link.

    Robust to None entries: close & navigate subtasks have ``subtask_objs[i] is
    None``; only open/close populate articulations. Never raises.
    """
    out = dict(
        active_obj=None,
        active_obj_merged=None,
        active_articulation=None,
        active_handle_link=None,
        active_handle_link_merged=None,
        active_subtask_type=None,
        active_obj_id=None,
    )
    try:
        ptr = int(_to_np(env.subtask_pointer)[env_idx])
    except Exception:
        return out

    task_plan = getattr(env, "task_plan", [])
    if not task_plan:
        return out
    ptr = min(ptr, len(task_plan) - 1)            # clip past-end pointer
    subtask = task_plan[ptr]
    out["active_subtask_type"] = _subtask_type(subtask)

    # Resolve the ORIGINAL obj_id from the un-merged task plan. MS-HAB rewrites
    # task_plan[ptr].obj_id to "obj_<num>" during _merge_pick_subtasks /
    # _merge_place_subtasks (mshab/envs/sequential_task.py:219, 274), so reading
    # subtask.obj_id directly would yield the merged name. The original plans
    # live in env.build_config_idx_to_task_plans, indexed per env.
    try:
        bcis = getattr(env, "build_config_idxs", None)
        tpis = getattr(env, "task_plan_idxs", None)
        bcitp = getattr(env, "build_config_idx_to_task_plans", None)
        if bcis is not None and tpis is not None and bcitp is not None:
            bci = int(bcis[env_idx])
            tpi = int(_to_np(tpis)[env_idx])
            tp_list = bcitp.get(bci) if hasattr(bcitp, "get") else None
            if tp_list is not None and 0 <= tpi < len(tp_list):
                original_plan = tp_list[tpi]
                subtasks = getattr(original_plan, "subtasks", None)
                if subtasks is not None and 0 <= ptr < len(subtasks):
                    out["active_obj_id"] = getattr(subtasks[ptr], "obj_id", None)
    except Exception:
        pass  # active_obj_id stays None; affordance lookup falls back to name.

    objs = getattr(env, "subtask_objs", [])
    if ptr < len(objs):
        out["active_obj"] = objs[ptr]             # may legitimately be None

    arts = getattr(env, "subtask_articulations", [])
    art = arts[ptr] if ptr < len(arts) else None
    out["active_articulation"] = art

    handle_idx = getattr(subtask, "articulation_handle_link_idx", None)
    if art is not None and handle_idx is not None:
        try:
            out["active_handle_link"] = art.links[handle_idx]
        except Exception:
            out["active_handle_link"] = None

    out["active_obj_merged"] = out["active_obj"]
    out["active_handle_link_merged"] = out["active_handle_link"]
    if object_name == "actual":
        seg_id_map = seg_id_map or {}
        out["active_obj"] = _resolve_actual_entity(
            out["active_obj"], seg_id_map, env_idx
        )
        out["active_handle_link"] = _resolve_actual_entity(
            out["active_handle_link"], seg_id_map, env_idx
        )
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def get_privileged_state(
    env,
    env_idx: int = 0,
    *,
    mshab_object_name: str = "actual",
) -> PrivilegedState:
    """Gather a typed privileged snapshot from a (possibly wrapped) env."""
    if mshab_object_name not in ("actual", "merged"):
        raise ValueError(
            "mshab_object_name must be 'actual' or 'merged', got "
            f"{mshab_object_name!r}"
        )
    e = env.unwrapped
    agent = e.agent
    scene = e.scene

    state = PrivilegedState(
        env=e,
        agent=agent,
        scene=scene,
        env_idx=env_idx,
        is_mshab=_looks_like_mshab(e),
        ee_links=get_ee_links(agent),
        tcp_pose_world=pose_to_world_array(get_tcp_pose(agent), env_idx),
        gripper_width=compute_gripper_width(agent, env_idx),
        seg_id_map=per_env_segmentation_id_map(e, env_idx),
        robot_links=_robot_link_set(agent),
    )

    if state.is_mshab:
        handles = _active_mshab_handles(
            e,
            env_idx,
            seg_id_map=state.seg_id_map,
            object_name=mshab_object_name,
        )
        state.active_obj = handles["active_obj"]
        state.active_obj_merged = handles["active_obj_merged"]
        state.active_articulation = handles["active_articulation"]
        state.active_handle_link = handles["active_handle_link"]
        state.active_handle_link_merged = handles["active_handle_link_merged"]
        state.active_subtask_type = handles["active_subtask_type"]
        state.active_obj_id = handles["active_obj_id"]
        if mshab_object_name == "merged":
            state.seg_id_map = _alias_segmentation_entity(
                state.seg_id_map, state.active_obj, env_idx
            )
            state.seg_id_map = _alias_segmentation_entity(
                state.seg_id_map, state.active_handle_link, env_idx
            )

    return state
