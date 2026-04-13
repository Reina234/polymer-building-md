from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroupType:
    name: str
    smarts: str
    overrides: frozenset[str] = field(default_factory=frozenset)
