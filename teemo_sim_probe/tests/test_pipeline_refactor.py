"""Regression tests for the simplified whitelist/segmentation pipeline."""

import unittest

import numpy as np

from teemo_sim_probe.adapters.privileged_state import clear_privileged_state_caches
from teemo_sim_probe.core.affordance import AffordanceComponent, AffordanceSet
from teemo_sim_probe.core.graph_builder import GraphBuilder
from teemo_sim_probe.core.node_builder import build_nodes
from teemo_sim_probe.core.schema import Edge, Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer
from teemo_sim_probe.core.entity_identity import stable_entity_key
from teemo_sim_probe.tools.build_subtask_whitelists import _WhitelistBuilder


class OneHopWhitelistTests(unittest.TestCase):
    def test_interactions_and_direct_support_only(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({
            "interacted": [
                {"key": "actor:024_bowl", "kind": "actor", "name": "bowl"},
                {"key": "link:cabinet/handle", "kind": "link", "name": "handle"},
            ],
            "supports": [
                {
                    "supporter": {
                        "key": "link:cabinet/drawer", "kind": "link",
                        "name": "drawer",
                    },
                    "supported_key": "actor:024_bowl",
                },
                # Recursive cabinet -> drawer evidence must be ignored because
                # drawer is a supporter, not an interacted root.
                {
                    "supporter": {
                        "key": "link:cabinet/body", "kind": "link",
                        "name": "body",
                    },
                    "supported_key": "link:cabinet/drawer",
                },
            ],
        })
        members = builder.payload()["members"]
        self.assertIn("actor:024_bowl", members)
        self.assertIn("link:cabinet/handle", members)
        self.assertIn("link:cabinet/drawer", members)
        self.assertNotIn("link:cabinet/body", members)
        self.assertEqual(members["link:cabinet/handle"]["roles"], ["interacted"])

    def test_uninteracted_handle_is_not_admitted(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({"interacted": [], "supports": []})
        self.assertNotIn("link:cabinet/handle", builder.payload()["members"])
        self.assertNotIn("actor:024_bowl", builder.payload()["members"])

    def test_supporter_requires_supported_interaction(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({
            "interacted": [],
            "supports": [{
                "supporter": {
                    "key": "link:cabinet/drawer", "kind": "link",
                    "name": "drawer",
                },
                "supported_key": "actor:024_bowl",
            }],
        })
        self.assertNotIn("link:cabinet/drawer", builder.payload()["members"])

    def test_interaction_types_track_contact_and_grasp(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({
            "interacted": [
                {"key": "actor:024_bowl", "kind": "actor", "name": "bowl",
                 "grasped": True},
                {"key": "link:cabinet/handle", "kind": "link", "name": "handle"},
            ],
            "supports": [{
                "supporter": {"key": "link:cabinet/drawer", "kind": "link",
                              "name": "drawer"},
                "supported_key": "actor:024_bowl",
            }],
        })
        members = builder.payload()["members"]
        # Bowl was grasped AND participated in a support pair as the supported
        # entity, so it carries contact + grasp + support tokens.
        self.assertEqual(
            members["actor:024_bowl"]["interaction_types"],
            ["contact", "grasp", "support"],
        )
        self.assertEqual(
            members["link:cabinet/handle"]["interaction_types"], ["contact"],
        )
        # Supporters now carry the ``support`` token so the runtime can gate
        # obj-obj support-compatibility on it.
        self.assertEqual(
            members["link:cabinet/drawer"]["interaction_types"], ["support"],
        )

    def test_bin_edges_emitted_from_bin_samples(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        # Repeat each sample so the 0.9 quantile equals the constant value.
        builder.absorb({
            "interacted": [
                {"key": "actor:024_bowl", "kind": "actor", "name": "bowl"},
            ],
            "supports": [],
            "bin_samples": {
                "planar_distance": [0.60] * 10,
                "height_offset": [0.30] * 10,
                "planar_distance_change": [0.10] * 10,
            },
        })
        payload = builder.payload()
        bins = payload["bin_edges"]
        # Equal-width 5-bin on [0, 0.60].
        self.assertEqual(len(bins["planar-distance"]), 4)
        self.assertAlmostEqual(bins["planar-distance"][0], 0.12, places=6)
        self.assertAlmostEqual(bins["planar-distance"][1], 0.24, places=6)
        self.assertAlmostEqual(bins["planar-distance"][2], 0.36, places=6)
        self.assertAlmostEqual(bins["planar-distance"][3], 0.48, places=6)
        # Signed 5-bin on |max|=0.30.
        self.assertEqual(len(bins["height-offset"]), 4)
        self.assertAlmostEqual(bins["height-offset"][0], -0.18, places=6)
        self.assertAlmostEqual(bins["height-offset"][1], -0.06, places=6)
        self.assertAlmostEqual(bins["height-offset"][2], 0.06, places=6)
        self.assertAlmostEqual(bins["height-offset"][3], 0.18, places=6)
        # Change relations use a narrower stable band: +/-v/10.
        self.assertAlmostEqual(bins["planar-distance-change"][1], -0.01, places=6)
        self.assertAlmostEqual(bins["planar-distance-change"][2], 0.01, places=6)
        # Compatibility absolute bins are always [1/3, 2/3].
        self.assertAlmostEqual(bins["grasp-compatibility"][0], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(bins["contact-compatibility"][1], 2.0 / 3.0, places=6)

    def test_compatibility_change_edges_mined_from_pose_samples(self):
        aff_set = AffordanceSet(by_object={
            "actor:024_bowl": [
                AffordanceComponent(
                    anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
                    preferred_width=0.05,
                ),
            ],
        })
        builder = _WhitelistBuilder(
            "pick",
            "actor:024_bowl",
            affordance_set=aff_set,
            temporal_k=1,
        )
        obj_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        builder.absorb({
            "interacted": [
                {
                    "key": "actor:024_bowl",
                    "kind": "actor",
                    "name": "bowl",
                    "grasped": True,
                    "max_ee_force": 1.0,
                },
            ],
            "supports": [],
            "bin_samples": {
                "planar_distance": [1.0] * 10,
                "height_offset": [0.5] * 10,
            },
            "pose_samples": [
                {
                    "tcp_pose": [0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "gripper_width": 0.05,
                    "entities": {
                        "actor:024_bowl": {
                            "pose": obj_pose,
                            "kind": "actor",
                            "name": "bowl",
                        },
                    },
                },
                {
                    "tcp_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "gripper_width": 0.05,
                    "entities": {
                        "actor:024_bowl": {
                            "pose": obj_pose,
                            "kind": "actor",
                            "name": "bowl",
                        },
                    },
                },
            ],
        })
        bins = builder.payload()["bin_edges"]
        self.assertIn("grasp-compatibility-change", bins)
        self.assertAlmostEqual(bins["grasp-compatibility-change"][1], -0.02)
        self.assertAlmostEqual(bins["grasp-compatibility-change"][2], 0.02)


class StaleEdgeTests(unittest.TestCase):
    def test_last_observed_edge_is_restored_as_stale(self):
        graph_builder = GraphBuilder.__new__(GraphBuilder)
        graph_builder._edge_history = {}
        ee = Node("ee", "ee", "end_effector")
        bowl = Node("actor:024_bowl", "object", "bowl")
        fresh = Graph(0, "env", "cam", nodes=[ee, bowl], edges=[
            Edge("ee", bowl.node_id, "contact", "contact", 2.0),
        ])
        graph_builder._attach_stale_edges(fresh, 0)

        stale_bowl = Node(
            bowl.node_id, "object", "bowl", visible=False,
            frozen_pose=True, persistent=True,
        )
        stale = Graph(3, "env", "cam", nodes=[ee, stale_bowl], edges=[])
        graph_builder._attach_stale_edges(stale, 3)
        self.assertEqual(len(stale.edges), 1)
        self.assertTrue(stale.edges[0].stale)
        self.assertEqual(stale.edges[0].observed_frame, 0)
        self.assertEqual(stale.edges[0].age, 3)

    def test_stale_edge_does_not_advance_temporal_history(self):
        # Use a continuous relation (planar-distance) since physical-state
        # predicates no longer feed the temporal buffer at all.
        buffer = TemporalBuffer(K=1)
        ee = Node("ee", "ee", "end_effector")
        bowl = Node("actor:024_bowl", "object", "bowl")
        fresh = Graph(0, "env", "cam", nodes=[ee, bowl], edges=[
            Edge("ee", bowl.node_id, "planar-distance", "near", 0.05),
        ])
        buffer.update(fresh)
        stale_bowl = Node(
            bowl.node_id, "object", "bowl", visible=False, frozen_pose=True,
        )
        stale = Graph(1, "env", "cam", nodes=[ee, stale_bowl], edges=[
            Edge(
                "ee", bowl.node_id, "planar-distance", "near", 0.10,
                stale=True, observed_frame=0, age=1,
            ),
        ])
        buffer.update(stale)
        # The fresh value gets purged when bowl becomes frozen-pose; the
        # stale-tagged edge is not re-ingested.
        self.assertNotIn(("ee", bowl.node_id, "planar-distance"), buffer._values)


class _Pose:
    p = np.array([[0.0, 0.0, 0.0]])
    q = np.array([[1.0, 0.0, 0.0, 0.0]])


class _Articulation:
    name = "env-0_cabinet-2"


class Link:
    def __init__(self, name):
        self.name = name
        self.pose = _Pose()
        self.articulation = _Articulation()


class _Actor:
    def __init__(self, name):
        self.name = name
        self.pose = _Pose()


# ``entity_kind`` dispatches on the literal class name.
_Actor.__name__ = "Actor"


class _State:
    def __init__(self, link=None, seg_id_map=None):
        self.env_idx = 0
        self.tcp_pose_world = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.seg_id_map = seg_id_map if seg_id_map is not None else {7: link}
        self.robot_links = set()
        self.ee_links = []


class SupporterMaskTests(unittest.TestCase):
    def test_visible_static_named_link_keeps_its_mask(self):
        link = Link("frl_apartment_drawer3")
        seg = np.array([[0, 7], [7, 7]], dtype=np.int32)
        nodes, masks, _cam, _rgb = build_nodes(
            {}, _State(link), seg_override=seg,
            rgb_override=np.zeros((2, 2, 3), dtype=np.uint8),
        )
        key = stable_entity_key(link)
        self.assertIn(key, nodes)
        self.assertEqual(masks.area(key), 3)
        self.assertEqual(key, "link:cabinet-2/frl_apartment_drawer3")


class WhitelistLoadCacheTests(unittest.TestCase):
    """The per-episode whitelist re-bind must not re-parse JSON from disk."""

    _PAYLOAD = {
        "_schema_version": 4,
        "subtask": "pick",
        "target": "actor:024_bowl",
        "members": {
            "actor:024_bowl": {
                "roles": ["interacted"],
                "interaction_types": ["contact"],
                "kind": "actor",
            },
        },
    }

    def _write(self, path, payload):
        import json
        with open(path, "w") as f:
            json.dump(payload, f)

    def test_unchanged_file_returns_cached_object(self):
        import os
        import tempfile
        from teemo_sim_probe.core.whitelist import load_whitelist

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pick_024_bowl.json")
            self._write(path, self._PAYLOAD)
            first = load_whitelist(path)
            second = load_whitelist(path)
            self.assertIs(first, second)

    def test_modified_file_is_reloaded(self):
        import copy
        import os
        import tempfile
        from teemo_sim_probe.core.whitelist import load_whitelist

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pick_024_bowl.json")
            self._write(path, self._PAYLOAD)
            first = load_whitelist(path)

            updated = copy.deepcopy(self._PAYLOAD)
            updated["members"]["link:cabinet/drawer"] = {
                "roles": ["support"],
                "interaction_types": ["support"],
                "kind": "link",
            }
            self._write(path, updated)
            st = os.stat(path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

            second = load_whitelist(path)
            self.assertIsNot(first, second)
            self.assertTrue(second.contains("link:cabinet/drawer"))
            self.assertFalse(first.contains("link:cabinet/drawer"))


class EntityMatchKeyTests(unittest.TestCase):
    """entity_match_key must resolve exactly like the node-level match_key."""

    def _assert_equivalent(self, entity):
        from teemo_sim_probe.core.node_builder import make_object_node
        from teemo_sim_probe.core.whitelist import entity_match_key, match_key

        node = make_object_node(entity, _State(entity))
        self.assertEqual(entity_match_key(entity), match_key(node))
        self.assertIsNotNone(entity_match_key(entity))

    def test_link_key_matches_node_key(self):
        self._assert_equivalent(Link("frl_apartment_drawer3"))

    def test_actor_key_matches_node_key(self):
        self._assert_equivalent(_Actor("env-0_024_bowl-3"))


class AdmitGateTests(unittest.TestCase):
    """The early admit gate must not change the post-apply_whitelist graph."""

    def _selector(self, whitelist):
        from teemo_sim_probe.core.selector import NodeSelector

        selector = NodeSelector({"selection": {"n_slots": 4, "k_persist": 0}})
        selector.set_whitelist(whitelist)
        return selector

    def test_gated_nodes_equal_ungated_after_whitelist(self):
        from teemo_sim_probe.core.node_builder import build_nodes
        from teemo_sim_probe.core.whitelist import Whitelist, entity_match_key

        admitted = Link("frl_apartment_drawer3")
        rejected = Link("frl_apartment_wall")
        state_kwargs = dict(seg_id_map={7: admitted, 9: rejected})
        seg = np.array([[0, 7], [9, 9]], dtype=np.int32)

        wl = Whitelist(
            subtask="pick",
            target="actor:024_bowl",
            by_key={entity_match_key(admitted): {"support"}},
            interaction_types={entity_match_key(admitted): {"support"}},
        )
        admit = lambda e: wl.contains(entity_match_key(e))

        gated, _, _, _ = build_nodes(
            {}, _State(**state_kwargs), seg_override=seg,
            rgb_override=np.zeros((2, 2, 3), dtype=np.uint8),
            need_masks=False, admit=admit,
        )
        ungated, _, _, _ = build_nodes(
            {}, _State(**state_kwargs), seg_override=seg,
            rgb_override=np.zeros((2, 2, 3), dtype=np.uint8),
            need_masks=False,
        )

        # The gate skips node construction for the never-admissible entity.
        self.assertNotIn(entity_match_key(rejected), gated)
        self.assertIn(entity_match_key(rejected), ungated)

        kept_gated = self._selector(wl).apply_whitelist(gated)
        kept_ungated = self._selector(wl).apply_whitelist(ungated)
        self.assertEqual(sorted(kept_gated), sorted(kept_ungated))
        for nid in kept_gated:
            self.assertEqual(
                kept_gated[nid].attributes.get("whitelist_roles"),
                kept_ungated[nid].attributes.get("whitelist_roles"),
            )
            self.assertEqual(
                kept_gated[nid].pixel_area, kept_ungated[nid].pixel_area,
            )

    def test_graph_builder_gate_admits_when_no_whitelist(self):
        from teemo_sim_probe.core.graph_builder import GraphBuilder

        builder = GraphBuilder.__new__(GraphBuilder)
        builder._match_key_cache = {}
        builder.selector = type("_S", (), {"whitelist": None})()
        self.assertTrue(builder._entity_admitted(Link("anything")))

    def test_graph_builder_gate_caches_match_key(self):
        from unittest import mock
        from teemo_sim_probe.core.graph_builder import GraphBuilder
        from teemo_sim_probe.core.whitelist import Whitelist, entity_match_key

        link = Link("frl_apartment_drawer3")
        key = entity_match_key(link)
        wl = Whitelist(by_key={key: {"support"}})

        builder = GraphBuilder.__new__(GraphBuilder)
        builder._match_key_cache = {}
        builder.selector = type("_S", (), {"whitelist": wl})()

        with mock.patch(
            "teemo_sim_probe.core.graph_builder.entity_match_key",
            side_effect=entity_match_key,
        ) as spy:
            self.assertTrue(builder._entity_admitted(link))
            self.assertTrue(builder._entity_admitted(link))
        self.assertEqual(spy.call_count, 1)


class LinkNameCacheTests(unittest.TestCase):
    def test_robot_link_names_cached_on_scene(self):
        from teemo_sim_probe.adapters.privileged_state import _robot_link_names

        calls = []

        class _Robot:
            def __init__(self):
                self.scene = type("_Scene", (), {})()

            def get_links(self):
                calls.append(1)
                return [Link("l_wheel_link"), Link("r_wheel_link")]

        agent = type("_Agent", (), {})()
        agent.robot = _Robot()

        first = _robot_link_names(agent)
        second = _robot_link_names(agent)
        self.assertIs(first, second)
        self.assertEqual(first, frozenset({"l_wheel_link", "r_wheel_link"}))
        self.assertEqual(len(calls), 1)


class _GraphEnvStub:
    def __init__(self, seg):
        self.unwrapped = type("_Base", (), {})()
        self.set_seg(seg)

    def set_seg(self, seg):
        self.unwrapped._last_obs = {
            "sensor_data": {
                "cam": {
                    "segmentation": seg,
                },
            },
        }


class GraphRuntimeMemoryTests(unittest.TestCase):
    def test_segmentation_cpu_buffer_is_reused(self):
        try:
            import torch
            from sac.graph_env import GraphObsBuilder
        except ImportError as exc:
            self.skipTest(f"torch-backed graph env unavailable: {exc}")

        builder = GraphObsBuilder.__new__(GraphObsBuilder)
        builder.env = _GraphEnvStub(
            torch.arange(8, dtype=torch.int32).reshape(2, 2, 2, 1)
        )
        builder.cameras = ["cam"]
        builder._cams_checked = False
        builder._seg_cpu_buffers = {}

        first = builder._read_batched_segs()
        self.assertTrue(
            np.array_equal(first["cam"], np.arange(8, dtype=np.int32).reshape(2, 2, 2))
        )
        ptr = builder._seg_cpu_buffers["cam"].data_ptr()

        builder.env.set_seg(
            torch.full((2, 2, 2, 1), 7, dtype=torch.int32)
        )
        second = builder._read_batched_segs()

        self.assertEqual(builder._seg_cpu_buffers["cam"].data_ptr(), ptr)
        self.assertTrue(np.array_equal(second["cam"], np.full((2, 2, 2), 7)))

    def test_scene_cache_cleanup_drops_teemo_caches(self):
        scene = type("_Scene", (), {})()
        scene._teemo_sidxs_cache = {"old": object()}
        scene._teemo_per_env_seg_maps = {0: {"old": object()}}
        scene._teemo_sliced_views = {(1, 0): object()}
        scene.keep_me = "still here"

        clear_privileged_state_caches(scene)

        self.assertNotIn("_teemo_sidxs_cache", scene.__dict__)
        self.assertNotIn("_teemo_per_env_seg_maps", scene.__dict__)
        self.assertNotIn("_teemo_sliced_views", scene.__dict__)
        self.assertEqual(scene.keep_me, "still here")


class PoseAccessorHotPathTests(unittest.TestCase):
    """Guard the .pose accessor call rate on the graph hot path."""

    def test_entity_pose_fires_property_once_per_frame(self):
        from teemo_sim_probe.adapters.privileged_state import (
            begin_frame_cache, end_frame_cache, entity_pose_world_array,
        )

        class _P:
            def __init__(self, p, q):
                self.p, self.q = p, q

        class _Entity:
            def __init__(self, batched_pose):
                self._batched_pose = batched_pose
                self._objs = [None] * len(batched_pose)
                self.pose_reads = 0

            @property
            def pose(self):
                self.pose_reads += 1
                b = self._batched_pose
                return _P(b[:, :3], b[:, 3:])

        num_envs = 8
        batched = np.tile(
            np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]), (num_envs, 1),
        )
        entity = _Entity(batched)

        begin_frame_cache()
        try:
            for env_idx in range(num_envs):
                out = entity_pose_world_array(entity, env_idx)
                self.assertIsNotNone(out)
                self.assertEqual(out.shape, (7,))
        finally:
            end_frame_cache()

        self.assertEqual(
            entity.pose_reads, 1,
            f"expected .pose to fire once per frame across {num_envs} envs, "
            f"got {entity.pose_reads}",
        )

    def test_get_tcp_pose_fires_property_once(self):
        from teemo_sim_probe.adapters.privileged_state import get_tcp_pose

        class _Agent:
            def __init__(self):
                self.tcp_pose_reads = 0
                self._value = object()

            @property
            def tcp_pose(self):
                self.tcp_pose_reads += 1
                return self._value

        agent = _Agent()
        result = get_tcp_pose(agent)
        self.assertIs(result, agent._value)
        self.assertEqual(
            agent.tcp_pose_reads, 1,
            f"expected 1 property fire, got {agent.tcp_pose_reads}",
        )


class RawBufferPoseFastPathTests(unittest.TestCase):
    """Guard the raw-rigid-body-buffer bypass of ManiSkill's ``.pose`` chain.

    The fast path is what actually eliminates the top allocator on the graph
    hot path (Link.pose / Actor.pose / cuda_rigid_body_data.torch() slice).
    These tests assert both correctness (the fast path returns the same rows
    the buffer would if indexed by hand) and, more importantly, that
    ``entity.pose`` and ``agent.tcp_pose`` are never fired when the fast
    path is active.
    """

    def _make_scene(self, buffer_np):
        class _Getter:
            def __init__(self, buf):
                self._buf = buf
            def torch(self_getter):
                class _T:
                    def __init__(self, arr):
                        self._arr = arr
                    def detach(_self):
                        return _self
                    def cpu(_self):
                        return _self
                    def numpy(_self):
                        return self_getter._buf
                return _T(self_getter._buf)

        class _Px:
            def __init__(self, buf):
                self.cuda_rigid_body_data = _Getter(buf)

        class _Scene:
            def __init__(self, buf):
                self.px = _Px(buf)
                self.parallel_in_single_scene = False

        return _Scene(buffer_np)

    def test_entity_pose_uses_fast_path_and_skips_accessor(self):
        from teemo_sim_probe.adapters.privileged_state import (
            begin_frame_cache, end_frame_cache, entity_pose_world_array,
        )

        num_envs = 4
        total_bodies = 10
        buf = np.zeros((total_bodies, 13), dtype=np.float32)
        for i in range(num_envs):
            buf[i + 3] = np.array(
                [i + 0.1, i + 0.2, i + 0.3, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]
            )

        class _Entity:
            def __init__(self):
                self._body_data_index = np.array(
                    [3, 4, 5, 6], dtype=np.int64,
                )
                self._objs = [None] * num_envs
                self.px_body_type = "dynamic"
                self.pose_reads = 0

            @property
            def pose(self):
                self.pose_reads += 1
                raise AssertionError(
                    ".pose must not fire when the raw-buffer fast path is active"
                )

        entity = _Entity()
        scene = self._make_scene(buf)

        begin_frame_cache(scene)
        try:
            for env_idx in range(num_envs):
                out = entity_pose_world_array(entity, env_idx)
                self.assertIsNotNone(out)
                expected = np.array(
                    [env_idx + 0.1, env_idx + 0.2, env_idx + 0.3,
                     1.0, 0.0, 0.0, 0.0]
                )
                np.testing.assert_allclose(out, expected, atol=1e-6)
        finally:
            end_frame_cache()

        self.assertEqual(
            entity.pose_reads, 0,
            "fast path leaked into the .pose accessor",
        )

    def test_static_entity_falls_back_to_pose_accessor(self):
        from teemo_sim_probe.adapters.privileged_state import (
            begin_frame_cache, end_frame_cache, entity_pose_world_array,
        )

        num_envs = 2
        buf = np.zeros((5, 13), dtype=np.float32)

        class _P:
            def __init__(self, p, q):
                self.p, self.q = p, q

        class _Entity:
            def __init__(self):
                self.px_body_type = "static"
                self._body_data_index = np.array([0, 1], dtype=np.int64)
                self._objs = [None] * num_envs
                self._batched = np.tile(
                    np.array([9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]), (num_envs, 1),
                )
                self.pose_reads = 0

            @property
            def pose(self):
                self.pose_reads += 1
                b = self._batched
                return _P(b[:, :3], b[:, 3:])

        entity = _Entity()
        scene = self._make_scene(buf)

        begin_frame_cache(scene)
        try:
            for env_idx in range(num_envs):
                out = entity_pose_world_array(entity, env_idx)
                self.assertIsNotNone(out)
                np.testing.assert_allclose(
                    out, np.array([9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]),
                )
        finally:
            end_frame_cache()

        self.assertEqual(
            entity.pose_reads, 1,
            "static entity should fall back to .pose exactly once per frame",
        )

    def test_tcp_pose_uses_fast_path_and_skips_accessor(self):
        from teemo_sim_probe.adapters.privileged_state import (
            begin_frame_cache, end_frame_cache, _tcp_pose_world_cached,
        )

        num_envs = 3
        buf = np.zeros((6, 13), dtype=np.float32)
        for i in range(num_envs):
            buf[i] = np.array(
                [0.5 + i, 0.6, 0.7, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]
            )

        class _TcpLink:
            def __init__(self):
                self._body_data_index = np.array([0, 1, 2], dtype=np.int64)
                self._objs = [None] * num_envs
                self.px_body_type = "dynamic"

        class _Agent:
            def __init__(self):
                self.tcp = _TcpLink()
                self.tcp_pose_reads = 0

            @property
            def tcp_pose(self):
                self.tcp_pose_reads += 1
                raise AssertionError(
                    "tcp_pose must not fire when the raw-buffer fast path is active"
                )

        agent = _Agent()
        scene = self._make_scene(buf)

        begin_frame_cache(scene)
        try:
            for env_idx in range(num_envs):
                out = _tcp_pose_world_cached(agent, env_idx)
                np.testing.assert_allclose(
                    out,
                    np.array([0.5 + env_idx, 0.6, 0.7, 1.0, 0.0, 0.0, 0.0]),
                    atol=1e-6,
                )
        finally:
            end_frame_cache()

        self.assertEqual(
            agent.tcp_pose_reads, 0,
            "fast path leaked into the tcp_pose accessor",
        )

    def test_parallel_in_single_scene_disables_fast_path(self):
        from teemo_sim_probe.adapters.privileged_state import (
            begin_frame_cache, end_frame_cache, entity_pose_world_array,
        )

        num_envs = 2
        buf = np.zeros((4, 13), dtype=np.float32)
        scene = self._make_scene(buf)
        scene.parallel_in_single_scene = True

        class _P:
            def __init__(self, p, q):
                self.p, self.q = p, q

        class _Entity:
            def __init__(self):
                self._body_data_index = np.array([0, 1], dtype=np.int64)
                self._objs = [None] * num_envs
                self.px_body_type = "dynamic"
                self._batched = np.tile(
                    np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]), (num_envs, 1),
                )
                self.pose_reads = 0

            @property
            def pose(self):
                self.pose_reads += 1
                b = self._batched
                return _P(b[:, :3], b[:, 3:])

        entity = _Entity()

        begin_frame_cache(scene)
        try:
            entity_pose_world_array(entity, 0)
            entity_pose_world_array(entity, 1)
        finally:
            end_frame_cache()

        self.assertEqual(
            entity.pose_reads, 1,
            "fast path must be disabled under parallel_in_single_scene "
            "and fall back through .pose exactly once per frame",
        )


if __name__ == "__main__":
    unittest.main()
