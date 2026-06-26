"""Identity-keyed slot assignment with reset_flag on identity change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SlotAssignment:
    slot_id: int
    reset_flag: bool


class SlotManager:
    def __init__(self, n_slots: int):
        self.n_slots = int(n_slots)
        # slot_id -> entity_id currently occupying it
        self._slot_to_entity: List[Optional[str]] = [None] * self.n_slots
        # entity_id -> slot_id
        self._entity_to_slot: Dict[str, int] = {}

    def reset_episode(self) -> None:
        self._slot_to_entity = [None] * self.n_slots
        self._entity_to_slot.clear()

    def assign(self, selected_entity_ids: List[str]) -> Dict[str, SlotAssignment]:
        """Sticky assignment. Returns {entity_id: SlotAssignment}.

        - Entities already in a slot keep that slot.
        - New entities take the lowest-index free slot, in the order given.
        - A slot reassigned to a different entity carries reset_flag=True.
        """
        selected = selected_entity_ids[: self.n_slots]
        assignments: Dict[str, SlotAssignment] = {}
        kept = set(selected)

        # Remember prior occupants BEFORE we clear vacated slots so we can
        # raise reset_flag when a new entity takes a slot whose previous
        # occupant just got evicted.
        prior_occupants = list(self._slot_to_entity)

        for s in range(self.n_slots):
            occupant = self._slot_to_entity[s]
            if occupant is not None and occupant not in kept:
                self._slot_to_entity[s] = None
                self._entity_to_slot.pop(occupant, None)

        unplaced: List[str] = []
        for ent_id in selected:
            s = self._entity_to_slot.get(ent_id)
            if s is not None:
                assignments[ent_id] = SlotAssignment(slot_id=s, reset_flag=False)
            else:
                unplaced.append(ent_id)

        free_slots = [s for s in range(self.n_slots)
                      if self._slot_to_entity[s] is None]
        for ent_id, slot in zip(unplaced, free_slots):
            prior = prior_occupants[slot]
            reset = (prior is not None and prior != ent_id)
            self._slot_to_entity[slot] = ent_id
            self._entity_to_slot[ent_id] = slot
            assignments[ent_id] = SlotAssignment(slot_id=slot, reset_flag=reset)

        return assignments

    def free_slots(self) -> List[int]:
        return [s for s in range(self.n_slots) if self._slot_to_entity[s] is None]

    def slot_of(self, entity_id: str) -> Optional[int]:
        return self._entity_to_slot.get(entity_id)
