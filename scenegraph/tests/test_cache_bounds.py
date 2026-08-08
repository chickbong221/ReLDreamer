"""Nothing that outlives an episode grows without bound.

Everything the builder owns is dropped by ``reset_episode``. What survives an
episode lives on the simulator's scene, keyed by ``id(entity)`` and holding the
entity beside the value, so each stale key pins a dead actor. The scene
signature only watches ``scene.actors``, which recreated merged actors never
enter, so a size cap is the only thing bounding those caches.
"""

import unittest
from types import SimpleNamespace

from scenegraph.adapters.graph_obs import GraphObsBuilder
from scenegraph.adapters.privileged_state import (
    _SCENE_CACHE_KEYS,
    clear_privileged_state_caches,
    purge_scene_caches,
    scene_cache_size,
)


def _env(**caches):
    scene = SimpleNamespace(**caches)
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)), scene


class SceneCacheBoundTests(unittest.TestCase):

    def test_the_size_is_the_largest_cache_not_the_total(self):
        env, _ = _env(_teemo_sidxs_cache={1: 'a', 2: 'b'},
                      _teemo_resolve_cache={3: 'c'})
        self.assertEqual(scene_cache_size(env), 2)

    def test_a_scene_that_has_not_cached_yet_reads_zero(self):
        env, _ = _env()
        self.assertEqual(scene_cache_size(env), 0)

    def test_below_the_cap_nothing_is_dropped(self):
        env, scene = _env(_teemo_sidxs_cache={1: 'a'})
        self.assertEqual(purge_scene_caches(env, cap=4), 1)
        self.assertEqual(scene._teemo_sidxs_cache, {1: 'a'})

    def test_past_the_cap_every_cache_goes_not_only_the_largest(self):
        # They key on the same recreated entities, so a stale entry in one is
        # a stale entry in all of them.
        env, scene = _env(
            _teemo_sidxs_cache={i: i for i in range(5)},
            _teemo_resolve_cache={1: 'c'},
        )
        self.assertEqual(purge_scene_caches(env, cap=4), 5)
        for key in _SCENE_CACHE_KEYS:
            self.assertNotIn(key, scene.__dict__)

    def test_a_reconfiguration_still_drops_everything_regardless_of_size(self):
        env, scene = _env(_teemo_sliced_views={1: 'a'}, _teemo_ee_links=[1])
        clear_privileged_state_caches(env)
        self.assertEqual(scene.__dict__, {})


class CacheEntriesMetricTests(unittest.TestCase):

    def _builder(self, stats):
        obj = object.__new__(GraphObsBuilder)
        obj.cache_stats = lambda: dict(stats)
        return obj

    def test_the_metric_totals_the_containers(self):
        b = self._builder({'registry': 3, 'appearance': 4})
        self.assertEqual(b.cache_entries, 7)

    def test_overflow_drops_is_not_a_container_size(self):
        # It counts vertices refused this episode, not entries being held.
        b = self._builder({'registry': 3, 'overflow_drops': 7})
        self.assertEqual(b.cache_entries, 3)


if __name__ == '__main__':
    unittest.main()
