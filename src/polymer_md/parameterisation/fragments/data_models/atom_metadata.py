from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtomMetadata:
    gaff2_type: str
    residue_id: str
    within_residue_position: int
