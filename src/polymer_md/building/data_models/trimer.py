from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from rdkit import Chem

TRIMER_REGION_TAG = "trimer_region"


class TrimerRegion(IntEnum):
    LEFT = 0
    CENTRAL = 1
    RIGHT = 2
    CAP = 3


@dataclass(frozen=True)
class BondType:
    from_site: int
    to_site: int

    @property
    def label(self) -> str:
        f = "H" if self.from_site == 0 else "T"
        t = "H" if self.to_site == 0 else "T"
        return f + t

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class TrimerResult:
    left_id: str
    central_id: str
    right_id: str
    left_bond: BondType
    right_bond: BondType
    mol: Chem.Mol = field(compare=False, hash=False)
    probability: float
    left_atom_indices: frozenset[int]
    central_atom_indices: frozenset[int]
    right_atom_indices: frozenset[int]
    cap_atom_indices: frozenset[int]

    @property
    def label(self) -> str:
        return (
            f"{self.left_id}[{self.left_bond}]"
            f"{self.central_id}[{self.right_bond}]"
            f"{self.right_id}"
        )
