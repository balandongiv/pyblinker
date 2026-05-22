"""Dictionary-oriented helpers for pyblinker utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, MutableMapping, Sequence, TypeVar

import pandas as pd

from pyblinker._logging import get_logger

logger = get_logger(__name__)


_T = TypeVar("_T")


def append_to_slot(slot: Any, value: _T) -> List[_T]:
    """Append ``value`` to ``slot`` preserving backwards compatible semantics."""

    if isinstance(slot, list):
        slot.append(value)
        return slot
    if pd.isna(slot):
        return [value]
    return [slot, value]


def contains_key(container: Mapping[str, Any] | pd.Series, key: str) -> bool:
    """Return ``True`` if ``key`` exists in ``container``."""

    if isinstance(container, pd.Series):
        return key in container.index
    return key in container


def group_by_key(
    items: Iterable[MutableMapping[str, Any]],
    key: str,
) -> Dict[Any, List[MutableMapping[str, Any]]]:
    """Group mapping items by ``key`` value."""

    grouped: Dict[Any, List[MutableMapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item[key]].append(item)
    return dict(grouped)


def update_dict_list(
    target: MutableMapping[str, List[_T]],
    key: str,
    value: Sequence[_T],
) -> None:
    """Ensure ``target[key]`` is a list and extend it with ``value``."""

    existing = list(target.get(key, []))
    existing.extend(value)
    target[key] = existing


__all__ = ["append_to_slot", "contains_key", "group_by_key", "update_dict_list"]
