from dataclasses import dataclass, field
from typing import Counter

from rdkit import Chem

from polymer_md.core.residue_instance import RESIDUE_TAG, ResidueInstance


@dataclass(frozen=True)
class Polymer:
    mol: Chem.rdchem.Mol = field(compare=False, hash=False)
    residue_instances: list[ResidueInstance]

    def atom_indices(self, instance: ResidueInstance) -> frozenset[int]:
        return instance.atom_indices(self.mol)

    def get_residue_by_atom_idx(self, atom_idx: int) -> ResidueInstance:
        atom = self.mol.GetAtomWithIdx(atom_idx)
        if not atom.HasProp(RESIDUE_TAG):
            raise ValueError(f"Atom {atom_idx} has no residue tag.")
        tag = atom.GetIntProp(RESIDUE_TAG)
        return next(r for r in self.residue_instances if tag in r.residue_tags)

    def composition(self) -> Counter[str]:
        return Counter(r.residue_id for r in self.residue_instances)

    def get_instances(self, residue_id: str) -> list[ResidueInstance]:
        return [r for r in self.residue_instances if r.residue_id == residue_id]
