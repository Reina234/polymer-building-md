from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto

from rdkit import Chem

TRIMER_REGION_TAG = "trimer_region"


class TrimerRegion(IntEnum):
    LEFT = 0
    CENTRAL = 1
    RIGHT = 2
    CAP = 3


class Orientation(Enum):
    HEAD_IN = auto()
    TAIL_IN = auto()


@dataclass(frozen=True)
class TrimerResult:
    left_id: str
    central_id: str
    right_id: str
    orientation: Orientation
    mol: Chem.Mol = field(compare=False, hash=False)
    probability: float
    left_atom_indices: frozenset[int]
    central_atom_indices: frozenset[int]
    right_atom_indices: frozenset[int]
    cap_atom_indices: frozenset[int]
