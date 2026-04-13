from dataclasses import dataclass
from typing import Counter, List

from rdkit import Chem

from polymer_md.core.residue_instance import ResidueInstance


@dataclass(frozen=True)
class Polymer:
    mol: Chem.rdchem.Mol
    residue_instances: List[ResidueInstance]

    def get_residue_by_atom(self, atom_idx: int) -> ResidueInstance:
        return next(r for r in self.residue_instances if atom_idx in r.atom_indices)

    def composition(self) -> Counter[str]:
        return Counter(r.residue_id for r in self.residue_instances)

    def get_instances(self, residue_id: str) -> list[ResidueInstance]:
        return [r for r in self.residue_instances if r.residue_id == residue_id]
