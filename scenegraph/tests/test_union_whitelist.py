"""Merging per-target whitelists into one file for multi-object runs.

The merge has to preserve two things the runtime depends on: every member from
every target, and the roles that decide what the selector does with them.
"""

import json
import tempfile
import unittest
from pathlib import Path

from scenegraph.tools.build_union_whitelist import merge


def _write(directory, subtask, target, members, robust=None):
    path = Path(directory) / f'{subtask}_{target}.json'
    path.write_text(json.dumps({
        '_schema_version': 4,
        'subtask': subtask,
        'target': f'actor:{target}',
        'members': members,
        'bin_stats_robust': robust or {},
        '_n_successful_rollouts': 30,
    }))
    return path


class MergeTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_members_from_every_target_survive(self):
        _write(self.dir, 'pick', '024_bowl', {
            'actor:024_bowl': {'roles': ['interacted'], 'kind': 'actor'},
            'link:counter/body': {'roles': ['support'], 'kind': 'link'},
        })
        _write(self.dir, 'pick', '013_apple', {
            'actor:013_apple': {'roles': ['interacted'], 'kind': 'actor'},
            'link:fridge/body': {'roles': ['support'], 'kind': 'link'},
        })
        out = merge(Path(self.dir), 'pick')
        self.assertEqual(set(out['members']), {
            'actor:024_bowl', 'actor:013_apple',
            'link:counter/body', 'link:fridge/body',
        })
        self.assertEqual(out['target'], 'all')

    def test_roles_union_rather_than_last_write_winning(self):
        # A link that only supports under one target and is also touched under
        # another has to keep both roles, or the selector reads it wrong.
        _write(self.dir, 'pick', 'a', {
            'link:counter/body': {'roles': ['support'], 'kind': 'link'}})
        _write(self.dir, 'pick', 'b', {
            'link:counter/body': {'roles': ['interacted'], 'kind': 'link'}})
        members = merge(Path(self.dir), 'pick')['members']
        self.assertEqual(
            members['link:counter/body']['roles'], ['interacted', 'support'])

    def test_supports_lists_accumulate(self):
        _write(self.dir, 'pick', 'a', {
            'link:counter/body': {
                'roles': ['support'], 'kind': 'link',
                'supports': ['actor:a']}})
        _write(self.dir, 'pick', 'b', {
            'link:counter/body': {
                'roles': ['support'], 'kind': 'link',
                'supports': ['actor:b']}})
        members = merge(Path(self.dir), 'pick')['members']
        self.assertEqual(
            members['link:counter/body']['supports'], ['actor:a', 'actor:b'])

    def test_bin_stats_take_the_widest_observation(self):
        # Bins are re-derived from these, so the merged file has to be
        # calibrated for the largest scene rather than whichever came first.
        _write(self.dir, 'pick', 'a', {'actor:a': {'roles': [], 'kind': 'actor'}},
               robust={'grasp': 1.0, 'contact': 5.0})
        _write(self.dir, 'pick', 'b', {'actor:b': {'roles': [], 'kind': 'actor'}},
               robust={'grasp': 3.0})
        out = merge(Path(self.dir), 'pick')
        self.assertEqual(out['bin_stats_robust'], {'grasp': 3.0, 'contact': 5.0})

    def test_a_previous_union_is_not_merged_into_itself(self):
        _write(self.dir, 'pick', '024_bowl', {
            'actor:024_bowl': {'roles': ['interacted'], 'kind': 'actor'}})
        _write(self.dir, 'pick', 'all', {
            'actor:stale': {'roles': ['interacted'], 'kind': 'actor'}})
        self.assertNotIn('actor:stale', merge(Path(self.dir), 'pick')['members'])

    def test_an_empty_directory_fails_loudly(self):
        with self.assertRaisesRegex(FileNotFoundError, 'pick_'):
            merge(Path(self.dir), 'pick')


class RolloutGroupingTests(unittest.TestCase):
    """One pkl can hold rollouts for many targets.

    The ``all`` policy picks a different object each episode, so the file-level
    entity_key names only whichever came first. Grouping on it would file every
    object's evidence under that one name.
    """

    def _pkl(self, directory, rollouts):
        import pickle
        path = Path(directory) / 'fetch' / 'pick'
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'all.pkl', 'wb') as handle:
            pickle.dump({
                '_schema_version': 7,
                'obj_id': 'all',
                # Only the first rollout's target, which is the whole point.
                'entity_key': rollouts[0]['target_key'],
                'subtask_type': 'pick',
                'interaction_rollouts': rollouts,
            }, handle)

    def test_one_mixed_pkl_produces_a_whitelist_per_target(self):
        from scenegraph.tools.build_subtask_whitelists import main

        states = tempfile.mkdtemp()
        out = Path(tempfile.mkdtemp())
        self._pkl(states, [
            {'target_key': 'actor:024_bowl', 'interacted': [
                {'key': 'actor:024_bowl', 'kind': 'actor', 'name': 'bowl'}]},
            {'target_key': 'actor:013_apple', 'interacted': [
                {'key': 'actor:013_apple', 'kind': 'actor', 'name': 'apple'}]},
        ])
        self.assertEqual(
            main(['--success-states-dir', states, '--out-dir', str(out)]), 0)
        written = sorted(p.name for p in out.glob('pick_*.json'))
        self.assertEqual(written, ['pick_013_apple.json', 'pick_024_bowl.json'])


if __name__ == '__main__':
    unittest.main()
