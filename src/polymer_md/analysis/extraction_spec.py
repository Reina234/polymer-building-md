from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
)
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
)


@dataclass(frozen=True)
class ExtractionSpec:
    pattern: str
    parameters: tuple[ForceFieldParameter, ...]

    def to_fragment(self) -> Fragment:
        mol = Chem.MolFromSmarts(self.pattern)
        if mol is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.pattern!r}")
        bond_params = [p for p in self.parameters if isinstance(p, BondParameter)]
        angle_params = [p for p in self.parameters if isinstance(p, AngleParameter)]
        dihedral_params = [p for p in self.parameters if isinstance(p, DihedralParameter)]
        atom_params = [p for p in self.parameters if isinstance(p, AtomParameter)]
        return Fragment(
            pattern=self.pattern,
            annotated_bonds=self._enumerate_bonds(mol, bond_params),
            annotated_angles=self._enumerate_angles(mol, angle_params),
            annotated_dihedrals=self._enumerate_dihedrals(mol, dihedral_params),
            annotated_atoms=self._enumerate_atoms(mol, atom_params),
        )

    @staticmethod
    def _enumerate_bonds(
        mol: Chem.Mol,
        parameters: list[BondParameter],
    ) -> tuple[AnnotatedBond, ...]:
        result = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            local_key = (min(i, j), max(i, j))
            for param in parameters:
                result.append(AnnotatedBond(local_indices=local_key, parameter=param))
        return tuple(result)

    @staticmethod
    def _enumerate_angles(
        mol: Chem.Mol,
        parameters: list[AngleParameter],
    ) -> tuple[AnnotatedAngle, ...]:
        result = []
        seen: set[tuple[int, int, int]] = set()
        for atom in mol.GetAtoms():
            j = atom.GetIdx()
            neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
            for pos_a, i in enumerate(neighbors):
                for k in neighbors[pos_a + 1:]:
                    key = (i, j, k)
                    if key not in seen:
                        seen.add(key)
                        for param in parameters:
                            result.append(AnnotatedAngle(local_indices=key, parameter=param))
        return tuple(result)

    @staticmethod
    def _enumerate_dihedrals(
        mol: Chem.Mol,
        parameters: list[DihedralParameter],
    ) -> tuple[AnnotatedDihedral, ...]:
        result = []
        seen: set[tuple[int, int, int, int]] = set()
        for bond in mol.GetBonds():
            j = bond.GetBeginAtomIdx()
            k = bond.GetEndAtomIdx()
            j_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
            k_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
            for i in j_neighbors:
                for l in k_neighbors:
                    if i == l:
                        continue
                    forward = (i, j, k, l)
                    reverse = (l, k, j, i)
                    key = min(forward, reverse)
                    if key not in seen:
                        seen.add(key)
                        for param in parameters:
                            result.append(AnnotatedDihedral(local_indices=key, parameter=param))
        return tuple(result)

    @staticmethod
    def _enumerate_atoms(
        mol: Chem.Mol,
        parameters: list[AtomParameter],
    ) -> tuple[AnnotatedAtom, ...]:
        result = []
        for atom in mol.GetAtoms():
            for param in parameters:
                result.append(AnnotatedAtom(local_index=atom.GetIdx(), parameter=param))
        return tuple(result)
