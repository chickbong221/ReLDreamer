"""The wrapper's stash is the graph's only handle on segmentation.

Upstream's ``FlattenRGBDObservationWrapper.observation`` pops ``sensor_data``
out of the dict it is handed, in place. Stashing the dict itself leaves the
graph reading an observation that has had its cameras removed.
"""

import unittest
from unittest import mock

try:
    from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper

    from embodied.envs.obs_wrappers import NamedCameraRGBWrapper
except ImportError:  # pragma: no cover - mani_skill is optional here
    NamedCameraRGBWrapper = None


class _Frame:
    """Stands in for a camera tensor; only ``clone`` is exercised."""

    def clone(self):
        return self


def _raw_obs():
    return {
        'agent': {},
        'sensor_data': {
            'fetch_head': {'rgb': _Frame(), 'segmentation': _Frame()},
            'fetch_hand': {'rgb': _Frame(), 'segmentation': _Frame()},
        },
    }


def _upstream(self, obs):
    """What upstream does to the dict: takes the cameras out of it."""
    obs.pop('sensor_data')
    return {'state': 'unchanged', 'rgb': 'concatenated'}


@unittest.skipIf(NamedCameraRGBWrapper is None, 'mani_skill is not installed')
class StashTests(unittest.TestCase):

    def setUp(self):
        self.wrapper = object.__new__(NamedCameraRGBWrapper)
        self.wrapper._camera_keys = {
            'image_head': 'fetch_head', 'image_hand': 'fetch_hand'}
        self.wrapper.raw_obs = None

    def _observe(self, obs):
        with mock.patch.object(
                FlattenRGBDObservationWrapper, 'observation',
                autospec=True, side_effect=_upstream):
            return self.wrapper.observation(obs)

    def test_the_stash_survives_upstream_emptying_the_observation(self):
        obs = _raw_obs()
        self._observe(obs)
        self.assertNotIn('sensor_data', obs)
        self.assertEqual(
            sorted(self.wrapper.raw_obs['sensor_data']),
            ['fetch_hand', 'fetch_head'])

    def test_the_stash_keeps_every_camera_field(self):
        self._observe(_raw_obs())
        head = self.wrapper.raw_obs['sensor_data']['fetch_head']
        self.assertEqual(sorted(head), ['rgb', 'segmentation'])

    def test_state_comes_straight_back_from_upstream(self):
        out = self._observe(_raw_obs())
        self.assertEqual(out['state'], 'unchanged')
        self.assertNotIn('rgb', out)
        self.assertEqual(sorted(out), ['image_hand', 'image_head', 'state'])

    def test_a_camera_the_obs_mode_omits_raises(self):
        obs = _raw_obs()
        obs['sensor_data'].pop('fetch_hand')
        with self.assertRaisesRegex(KeyError, 'fetch_hand'):
            self._observe(obs)


if __name__ == '__main__':
    unittest.main()
