from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class MolAtom:
    mol: Chem.rdchem.Mol
    idx: int

    @property
    def atom(self) -> Chem.rdchem.Atom:
        return self.mol.GetAtomWithIdx(self.idx)
