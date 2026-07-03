"""Closed vocabularies for the graph encoder.

Node vocab: whitelist-key union scanned from every per-subtask whitelist under
``whitelist_dir``, plus ``<ee>`` and ``<pad>``.

Edge vocab: flat ``(relation, label)`` pairs enumerated from relation_rules
(spatial + compat + change + physical-state singletons), plus ``<pad_edge>``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core.relation_rules import (
    CHANGE_LABELS,
    COMPAT_LABELS,
    SPATIAL_LABELS,
)


PAD_TOKEN = "<pad>"
EE_TOKEN = "<ee>"
PAD_EDGE_TOKEN = "<pad_edge>"

_PHYSICAL_STATE = ("contact", "grasp", "support", "contain")


@dataclass
class NodeVocab:
    token_to_id: Dict[str, int]

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def ee_id(self) -> int:
        return self.token_to_id[EE_TOKEN]

    def encode(self, key: Optional[str]) -> int:
        if key is None:
            return self.pad_id
        idx = self.token_to_id.get(key)
        if idx is None:
            raise KeyError(
                f"NodeVocab: unknown node key {key!r}. Vocab must be built "
                f"from a whitelist directory that covers every runtime asset."
            )
        return idx


@dataclass
class EdgeVocab:
    token_to_id: Dict[Tuple[str, str], int]

    def __len__(self) -> int:
        return len(self.token_to_id) + 1  # +1 for pad

    @property
    def pad_id(self) -> int:
        return 0

    def encode(self, relation: str, label: str) -> int:
        idx = self.token_to_id.get((relation, label))
        if idx is None:
            raise KeyError(
                f"EdgeVocab: unknown (relation, label)=({relation!r}, {label!r})"
            )
        return idx


def build_edge_vocab() -> EdgeVocab:
    token_to_id: Dict[Tuple[str, str], int] = {}
    next_id = 1  # 0 reserved for pad
    for rel, labels in SPATIAL_LABELS.items():
        for lab in labels:
            token_to_id[(rel, lab)] = next_id
            next_id += 1
    for rel, labels in COMPAT_LABELS.items():
        for lab in labels:
            token_to_id[(rel, lab)] = next_id
            next_id += 1
    for rel, labels in CHANGE_LABELS.items():
        for lab in labels:
            token_to_id[(rel, lab)] = next_id
            next_id += 1
    for rel in _PHYSICAL_STATE:
        token_to_id[(rel, rel)] = next_id
        next_id += 1
    return EdgeVocab(token_to_id=token_to_id)


def build_node_vocab(whitelist_dir: str) -> NodeVocab:
    if not os.path.isdir(whitelist_dir):
        raise FileNotFoundError(
            f"whitelist_dir does not exist: {whitelist_dir!r}. "
            "Mine assets with tools/build_subtask_whitelists.py first."
        )
    keys: List[str] = [PAD_TOKEN, EE_TOKEN]
    seen = set(keys)
    for name in sorted(os.listdir(whitelist_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(whitelist_dir, name)
        with open(path, "r") as f:
            raw = json.load(f)
        members = raw.get("members", {}) if isinstance(raw, dict) else {}
        for member_key in members.keys():
            if not isinstance(member_key, str) or member_key.startswith("_"):
                continue
            if member_key not in seen:
                seen.add(member_key)
                keys.append(member_key)
    if len(keys) <= 2:
        raise ValueError(
            f"whitelist_dir {whitelist_dir!r} contained no member keys"
        )
    return NodeVocab(token_to_id={k: i for i, k in enumerate(keys)})


def node_key_for(node) -> Optional[str]:
    """Vocab key for one graph node. ``None`` for padding."""
    if not getattr(node, "valid_mask", True):
        return None
    if node.node_type == "ee":
        return EE_TOKEN
    return node.attributes.get("whitelist_key")
