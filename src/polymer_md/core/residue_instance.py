from dataclasses import dataclass
from enum import Enum, auto


class ResidueType(Enum):
    MONOMER = auto()
    CAP = auto()


@dataclass(frozen=True)
class ResidueInstance:
    residue_id: str
    instance_number: int
    atom_indices: frozenset[int]
    residue_type: ResidueType = ResidueType.MONOMER
