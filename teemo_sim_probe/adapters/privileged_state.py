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

    # Segmentation id -> Actor/Link (ManiSkill primitive).
    seg_id_map: Dict[int, Any] = field(default_factory=dict)

    # Robot link set (for "is this a robot link?" tests).
    robot_links: set = field(default_factory=set)

    # MS-HAB active manipulation handles for this env_idx (any may be None).
    active_obj: Optional[Any] = None
    active_articulation: Optional[Any] = None
    active_handle_link: Optional[Any] = None
    active_subtask_type: Optional[str] = None

    # ----- queries the relation rules call ------------------------------- #
    def pairwise_force(self, a: Any, b: Any) -> float:
        """Scalar contact-force magnitude between two entities for this env."""
        if a is None or b is None:
            return 0.0
        f = self.scene.get_pairwise_contact_forces(a, b)      # [num_envs, 3]
        return float(np.linalg.norm(_to_np(f)[self.env_idx]))

    def ee_object_contact_force(self, obj: Any) -> float:
        """Sum of both finger contact forces against ``obj`` (MS-HAB style)."""
        if obj is None:
            return 0.0
        f1 = self.pairwise_force(self.agent.finger1_link, obj)
        f2 = self.pairwise_force(self.agent.finger2_link, obj)
        return f1 + f2

    def is_grasping(self, obj: Any, max_angle: int = 30) -> bool:
        if obj is None or not hasattr(self.agent, "is_grasping"):
            return False
        try:
            g = self.agent.is_grasping(obj, max_angle=max_angle)
        except TypeError:
            g = self.agent.is_grasping(obj)
        return bool(_to_np(g)[self.env_idx])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def get_tcp_pose(agent) -> Any:
    """Fetch defines ``tcp_pose`` (computed); Panda exposes ``tcp.pose``."""
    if hasattr(agent, "tcp_pose"):
        return agent.tcp_pose
    return agent.tcp.pose


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


def _active_mshab_handles(env, env_idx: int) -> Dict[str, Any]:
    """Resolve the current subtask's object / articulation / handle link.

    Robust to None entries: close & navigate subtasks have ``subtask_objs[i] is
    None``; only open/close populate articulations. Never raises.
    """
    out = dict(
        active_obj=None,
        active_articulation=None,
        active_handle_link=None,
        active_subtask_type=None,
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
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def get_privileged_state(env, env_idx: int = 0) -> PrivilegedState:
    """Gather a typed privileged snapshot from a (possibly wrapped) env."""
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
        seg_id_map=dict(getattr(e, "segmentation_id_map", {})),
        robot_links=_robot_link_set(agent),
    )

    if state.is_mshab:
        handles = _active_mshab_handles(e, env_idx)
        state.active_obj = handles["active_obj"]
        state.active_articulation = handles["active_articulation"]
        state.active_handle_link = handles["active_handle_link"]
        state.active_subtask_type = handles["active_subtask_type"]

    return state
