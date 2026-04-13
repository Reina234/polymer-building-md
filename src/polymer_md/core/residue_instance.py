from dataclasses import dataclass
from enum import Enum, auto

from rdkit import Chem

RESIDUE_TAG = "residue_instance_tag"


class ResidueType(Enum):
    MONOMER = auto()
    CAP = auto()


@dataclass(frozen=True)
class ResidueInstance:
    residue_id: str
    instance_number: int
    residue_tags: frozenset[int]
    residue_type: ResidueType

    def atom_indices(self, mol: Chem.rdchem.Mol) -> frozenset[int]:
        return frozenset(
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.HasProp(RESIDUE_TAG) and a.GetIntProp(RESIDUE_TAG) in self.residue_tags
        )
