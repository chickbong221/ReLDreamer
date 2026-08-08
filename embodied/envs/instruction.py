"""Frozen instruction embeddings, read per env from the active subtask.

The vector is a property of the task rather than of the pixels: it is computed
once offline by ``scenegraph/tools/build_instruction_embeddings.py`` and only
looked up here, so nothing the agent trains can move it and the stored value
stays valid for a whole run. Every method loads the same table, which keeps the
instruction channel identical across a comparison.

MS-HAB hands a different object to each episode and advances
``subtask_pointer`` within one, so the lookup runs every step rather than once
per reset. It reads the pointer and the un-merged task plans directly:
``env.get_obs()`` would rerun ``evaluate`` and mutate that pointer.
"""

import numpy as np


def _to_np(value):
    """Batched handle -> 1-D numpy, whatever container MS-HAB used."""
    if value is None:
        return None
    cpu = getattr(value, 'cpu', None)
    if cpu is not None:
        value = cpu()
    return np.asarray(value).reshape(-1)


class InstructionTable:
    """``<subtask>/<target>`` -> one frozen embedding row."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.vectors = np.asarray(data['vectors'], np.float32)
        self.keys = [str(k) for k in data['keys']]
        self.model = str(data['model'])
        if self.vectors.ndim != 2 or len(self.keys) != len(self.vectors):
            raise ValueError(
                f'instruction table {path!r} is malformed: {len(self.keys)} '
                f'keys against vectors {self.vectors.shape}')
        self._rows = {k: i for i, k in enumerate(self.keys)}

    @property
    def dim(self):
        return int(self.vectors.shape[1])

    def row(self, subtask, target):
        key = f'{subtask}/{target}'
        index = self._rows.get(key)
        if index is None:
            raise KeyError(
                f'instruction table has no entry for {key!r}; rebuild it over '
                'every split this run touches (model='
                f'{self.model}, {len(self.keys)} entries)')
        return self.vectors[index]


class InstructionReader:
    """Per-env instruction rows for whichever subtask is currently active."""

    def __init__(self, env, table, num_envs):
        from scenegraph.core.affordance import canonical_affordance_key

        self.table = table
        self.num_envs = int(num_envs)
        self._base = env.unwrapped
        self._canonical = canonical_affordance_key
        self._last = np.zeros((self.num_envs, table.dim), np.float32)
        # (build config, task plan, pointer) -> row. The triple changes only
        # when a subtask advances, so the regex and the dict lookup do not run
        # every step for every env.
        self._resolved = {}

    def step(self, is_last=None):
        """``[num_envs, dim]`` for this frame.

        Envs flagged in ``is_last`` re-emit their previous row: the vector env
        auto-resets inside ``step``, so their pointer already belongs to the
        next episode.
        """
        base = self._base
        ptrs = _to_np(getattr(base, 'subtask_pointer', None))
        tpis = _to_np(getattr(base, 'task_plan_idxs', None))
        bcis = _to_np(getattr(base, 'build_config_idxs', None))
        plans = getattr(base, 'build_config_idx_to_task_plans', None)
        if ptrs is None or tpis is None or bcis is None or plans is None:
            raise RuntimeError(
                'instruction: this env exposes no MS-HAB task plan '
                '(subtask_pointer / task_plan_idxs / build_config_idxs / '
                'build_config_idx_to_task_plans), so there is no subtask to '
                'name; the instruction input needs an MS-HAB task')
        done = (
            np.zeros(self.num_envs, bool) if is_last is None
            else np.asarray(is_last, bool).reshape(-1))
        for i in range(self.num_envs):
            if done[i]:
                continue
            triple = (int(bcis[i]), int(tpis[i]), int(ptrs[i]))
            row = self._resolved.get(triple)
            if row is None:
                row = self.table.row(*self._active(plans, triple))
                self._resolved[triple] = row
            self._last[i] = row
        return self._last.copy()

    def _active(self, plans, triple):
        """The ``(subtask, target)`` pair naming one env's current objective.

        Reads the un-merged plans: MS-HAB rewrites ``task_plan[ptr].obj_id`` to
        ``obj_<num>`` while merging subtasks, and the original name is what the
        table is keyed on.
        """
        bci, tpi, ptr = triple
        group = plans.get(bci) if hasattr(plans, 'get') else plans[bci]
        subtasks = getattr(group[tpi], 'subtasks', None) or []
        if not subtasks:
            raise RuntimeError(
                f'instruction: task plan {bci}/{tpi} carries no subtasks')
        # The pointer runs one past the end once every subtask is done.
        subtask = subtasks[min(ptr, len(subtasks) - 1)]
        kind = str(getattr(subtask, 'type', ''))
        obj_id = getattr(subtask, 'obj_id', None)
        if not obj_id:
            raise RuntimeError(
                f'instruction: subtask {kind!r} carries no obj_id, so it names '
                'no object target')
        return kind, 'actor:' + str(self._canonical(str(obj_id)))
