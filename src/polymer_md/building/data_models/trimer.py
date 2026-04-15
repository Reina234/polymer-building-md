from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from rdkit import Chem


class Orientation(Enum):
    HEAD_IN = auto()
    TAIL_IN = auto()


@dataclass(frozen=True)
class TrimerResult:
    left_id: str
    central_id: str
    right_id: str
    orientation: Orientation
    mol: Chem.Mol
    probability: float
