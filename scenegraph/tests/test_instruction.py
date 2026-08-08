"""The instruction channel: table lookup, per-env resolution, holdout split.

The vector is the one input every method in the comparison receives
identically, so the failures that matter are the quiet ones -- a wrong row for
an env, a stale row on a terminal frame, or a holdout that silently kept the
categories it was meant to remove.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from embodied.envs.instruction import InstructionReader, InstructionTable
from embodied.envs.maniskill import _plan_objects, _use_named_cameras


def _table(tmp, keys, dim=4, model='stub'):
    vectors = np.arange(len(keys) * dim, dtype=np.float32).reshape(len(keys), dim)
    np.savez(tmp, keys=np.array(keys), vectors=vectors, model=np.array(model))
    return InstructionTable(tmp)


def _subtask(kind='pick', obj_id='024_bowl-3'):
    return SimpleNamespace(type=kind, obj_id=obj_id)


def _plan(*subtasks):
    return SimpleNamespace(subtasks=list(subtasks))


def _env(plans, ptrs, tpis, bcis):
    return SimpleNamespace(unwrapped=SimpleNamespace(
        build_config_idx_to_task_plans=plans,
        subtask_pointer=np.asarray(ptrs),
        task_plan_idxs=np.asarray(tpis),
        build_config_idxs=np.asarray(bcis),
    ))


class TableTests(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        self.tmp.close()

    def test_a_row_comes_back_by_subtask_and_target(self):
        table = _table(self.tmp.name, ['pick/actor:024_bowl', 'pick/actor:013_apple'])
        np.testing.assert_allclose(
            table.row('pick', 'actor:013_apple'), [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(table.dim, 4)

    def test_a_missing_entry_names_the_key_it_wanted(self):
        table = _table(self.tmp.name, ['pick/actor:024_bowl'])
        with self.assertRaisesRegex(KeyError, 'pick/actor:013_apple'):
            table.row('pick', 'actor:013_apple')

    def test_a_key_count_that_disagrees_with_the_vectors_is_rejected(self):
        np.savez(
            self.tmp.name, keys=np.array(['a', 'b']),
            vectors=np.zeros((1, 4), np.float32), model=np.array('stub'))
        with self.assertRaisesRegex(ValueError, 'malformed'):
            InstructionTable(self.tmp.name)


class ReaderTests(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        self.tmp.close()
        self.table = _table(
            self.tmp.name, ['pick/actor:024_bowl', 'pick/actor:013_apple'])
        self.bowl = self.table.row('pick', 'actor:024_bowl')
        self.apple = self.table.row('pick', 'actor:013_apple')

    def _reader(self, env, num_envs):
        return InstructionReader(env, self.table, num_envs)

    def test_each_environment_gets_the_object_its_own_plan_names(self):
        # pick_all hands a different object to each episode, so two envs
        # running at once are usually on different targets.
        env = _env(
            {0: [_plan(_subtask(obj_id='024_bowl-3')),
                 _plan(_subtask(obj_id='013_apple-0'))]},
            ptrs=[0, 0], tpis=[0, 1], bcis=[0, 0])
        out = self._reader(env, 2).step()
        np.testing.assert_allclose(out[0], self.bowl)
        np.testing.assert_allclose(out[1], self.apple)

    def test_advancing_the_pointer_changes_the_instruction(self):
        env = _env(
            {0: [_plan(_subtask(obj_id='024_bowl-3'),
                       _subtask(obj_id='013_apple-0'))]},
            ptrs=[0], tpis=[0], bcis=[0])
        reader = self._reader(env, 1)
        np.testing.assert_allclose(reader.step()[0], self.bowl)
        env.unwrapped.subtask_pointer = np.asarray([1])
        np.testing.assert_allclose(reader.step()[0], self.apple)

    def test_a_terminal_frame_keeps_the_row_it_had(self):
        # The vector env auto-resets inside step(), so a done env's pointer
        # already belongs to the next episode.
        env = _env(
            {0: [_plan(_subtask(obj_id='024_bowl-3'),
                       _subtask(obj_id='013_apple-0'))]},
            ptrs=[0], tpis=[0], bcis=[0])
        reader = self._reader(env, 1)
        reader.step()
        env.unwrapped.subtask_pointer = np.asarray([1])
        np.testing.assert_allclose(reader.step(is_last=[True])[0], self.bowl)

    def test_a_pointer_past_the_last_subtask_reads_the_last_one(self):
        env = _env(
            {0: [_plan(_subtask(obj_id='024_bowl-3'))]},
            ptrs=[7], tpis=[0], bcis=[0])
        np.testing.assert_allclose(self._reader(env, 1).step()[0], self.bowl)

    def test_an_environment_without_a_task_plan_fails_loudly(self):
        env = SimpleNamespace(unwrapped=SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, 'subtask_pointer'):
            self._reader(env, 1).step()

    def test_a_subtask_with_no_object_fails_loudly(self):
        env = _env(
            {0: [_plan(_subtask(kind='open', obj_id=None))]},
            ptrs=[0], tpis=[0], bcis=[0])
        with self.assertRaisesRegex(RuntimeError, 'obj_id'):
            self._reader(env, 1).step()


class HoldoutTests(unittest.TestCase):

    def test_plan_objects_are_canonical_not_per_instance(self):
        # The holdout list, the whitelist filenames and the instruction keys
        # all use the canonical form, so the filter has to as well.
        plan = _plan(_subtask(obj_id='env-0_024_bowl-3'),
                     _subtask(obj_id='013_apple-0'))
        self.assertEqual(_plan_objects(plan), {'024_bowl', '013_apple'})

    def test_a_plan_naming_no_object_contributes_nothing(self):
        self.assertEqual(_plan_objects(_plan(_subtask(obj_id=None))), set())

    def test_the_split_is_complementary(self):
        # Whatever training drops is exactly what evaluation keeps: a plan
        # counted by both would put a held-out category into pretraining.
        holdout = {'024_bowl'}
        plans = [
            _plan(_subtask(obj_id='024_bowl-3')),
            _plan(_subtask(obj_id='013_apple-0')),
            _plan(_subtask(obj_id='024_bowl-1'), _subtask(obj_id='013_apple-2')),
        ]
        keep = lambda want: [
            p for p in plans if bool(_plan_objects(p) & holdout) == want]
        excluded, only = keep(False), keep(True)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(len(only), 2)
        self.assertEqual(len(excluded) + len(only), len(plans))
        for plan in excluded:
            self.assertNotIn('024_bowl', _plan_objects(plan))


class NamedCameraFlagTests(unittest.TestCase):
    """The baseline arm has to see the pixels the graph arm sees."""

    def test_auto_follows_the_graph(self):
        self.assertTrue(_use_named_cameras('auto', {'enabled': True}))
        self.assertFalse(_use_named_cameras('auto', {'enabled': False}))

    def test_pinning_it_survives_the_graph_being_off(self):
        # Without this the graph-free run silently switches to one
        # channel-concatenated image and stops being a controlled comparison.
        self.assertTrue(_use_named_cameras('true', {'enabled': False}))
        self.assertTrue(_use_named_cameras(True, {'enabled': False}))

    def test_the_string_false_is_not_read_as_truthy(self):
        # elements.Config hands CLI overrides through as strings, and bool()
        # of a non-empty string is True.
        self.assertFalse(_use_named_cameras('false', {'enabled': True}))
        self.assertFalse(_use_named_cameras('False', {'enabled': True}))

    def test_an_unrecognised_setting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'named_cameras'):
            _use_named_cameras('sometimes', {'enabled': True})


if __name__ == '__main__':
    unittest.main()
