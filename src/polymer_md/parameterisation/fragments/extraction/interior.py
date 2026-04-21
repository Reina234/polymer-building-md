from __future__ import annotations

from dataclasses import dataclass

import parmed as pmd
from rdkit import Chem

from polymer_md.building.data_models.trimer import TrimerResult
from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
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
)
from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder
from polymer_md.utils.parmed_helper import CoordinateCrosswalk, StructureMolDeriver


@dataclass
class InteriorFragmentExtractor:
    def extract(self, parameterised_trimer: ParameterisedTrimer) -> list[Fragment]:
        derived_mol = StructureMolDeriver.derive(parameterised_trimer.structure)
        mol3d_to_parmed = CoordinateCrosswalk.map_mol3d_to_parmed(
            parameterised_trimer.mol_3d,
            parameterised_trimer.structure,
        )
        central_parmed_indices = self._resolve_central_parmed_indices(
            parameterised_trimer.trimer_result,
            parameterised_trimer.mol_3d,
            mol3d_to_parmed,
        )
        return (
            self._extract_atom_fragments(derived_mol, central_parmed_indices)
            + self._extract_bond_fragments(derived_mol, parameterised_trimer.structure, central_parmed_indices)
            + self._extract_angle_fragments(derived_mol, parameterised_trimer.structure, central_parmed_indices)
            + self._extract_dihedral_fragments(derived_mol, parameterised_trimer.structure, central_parmed_indices)
        )

    def _resolve_central_parmed_indices(
        self,
        trimer_result: TrimerResult,
        mol_3d: Chem.Mol,
        mol3d_to_parmed: dict[int, int],
    ) -> frozenset[int]:
        central_heavy_mol3d = trimer_result.central_atom_indices
        central_hydrogen_mol3d = self._find_central_hydrogen_indices(mol_3d, central_heavy_mol3d)
        all_central_mol3d = central_heavy_mol3d | central_hydrogen_mol3d
        return frozenset(
            mol3d_to_parmed[index]
            for index in all_central_mol3d
            if index in mol3d_to_parmed
        )

    def _find_central_hydrogen_indices(
        self,
        mol_3d: Chem.Mol,
        central_heavy_indices: frozenset[int],
    ) -> frozenset[int]:
        central_hydrogens = set()
        for atom in mol_3d.GetAtoms():
            if not self._is_hydrogen(atom):
                continue
            parent_index = atom.GetNeighbors()[0].GetIdx()
            if parent_index in central_heavy_indices:
                central_hydrogens.add(atom.GetIdx())
        return frozenset(central_hydrogens)

    def _is_hydrogen(self, atom: Chem.Atom) -> bool:
        return atom.GetAtomicNum() == 1

    def _extract_atom_fragments(
        self,
        derived_mol: Chem.Mol,
        central_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        return [
            self._build_atom_fragment(derived_mol, atom_index)
            for atom_index in central_parmed_indices
        ]

    def _extract_bond_fragments(
        self,
        derived_mol: Chem.Mol,
        structure: pmd.Structure,
        central_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        seen_keys: set[tuple[int, int]] = set()
        fragments = []
        for bond in structure.bonds:
            atom_i, atom_j = bond.atom1.idx, bond.atom2.idx
            if not self._bond_touches_central(atom_i, atom_j, central_parmed_indices):
                continue
            key = self._bond_key(atom_i, atom_j)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if derived_mol.GetBondBetweenAtoms(atom_i, atom_j) is None:
                continue
            fragments.append(self._build_bond_fragment(derived_mol, atom_i, atom_j))
        return fragments

    def _extract_angle_fragments(
        self,
        derived_mol: Chem.Mol,
        structure: pmd.Structure,
        central_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        seen_keys: set[tuple[int, int, int]] = set()
        fragments = []
        for angle in structure.angles:
            atom_i, atom_j, atom_k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
            if atom_j not in central_parmed_indices:
                continue
            key = self._angle_key(atom_i, atom_j, atom_k)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fragments.append(self._build_angle_fragment(derived_mol, atom_i, atom_j, atom_k))
        return fragments

    def _extract_dihedral_fragments(
        self,
        derived_mol: Chem.Mol,
        structure: pmd.Structure,
        central_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        seen_keys: set[tuple[int, int, int, int]] = set()
        fragments = []
        for dihedral in structure.dihedrals:
            if dihedral.improper:
                continue
            atom_i = dihedral.atom1.idx
            atom_j = dihedral.atom2.idx
            atom_k = dihedral.atom3.idx
            atom_l = dihedral.atom4.idx
            if not self._dihedral_middle_touches_central(atom_j, atom_k, central_parmed_indices):
                continue
            key = self._dihedral_key(atom_i, atom_j, atom_k, atom_l)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fragments.append(self._build_dihedral_fragment(derived_mol, *key))
        return fragments

    def _bond_touches_central(
        self,
        atom_i: int,
        atom_j: int,
        central: frozenset[int],
    ) -> bool:
        return atom_i in central or atom_j in central

    def _dihedral_middle_touches_central(
        self,
        atom_j: int,
        atom_k: int,
        central: frozenset[int],
    ) -> bool:
        return atom_j in central or atom_k in central

    def _bond_key(self, atom_i: int, atom_j: int) -> tuple[int, int]:
        return (min(atom_i, atom_j), max(atom_i, atom_j))

    def _angle_key(self, atom_i: int, atom_j: int, atom_k: int) -> tuple[int, int, int]:
        return (min(atom_i, atom_k), atom_j, max(atom_i, atom_k))

    def _dihedral_key(
        self,
        atom_i: int,
        atom_j: int,
        atom_k: int,
        atom_l: int,
    ) -> tuple[int, int, int, int]:
        forward = (atom_i, atom_j, atom_k, atom_l)
        reverse = (atom_l, atom_k, atom_j, atom_i)
        return min(forward, reverse)

    def _build_atom_fragment(self, derived_mol: Chem.Mol, atom_index: int) -> Fragment:
        pattern = SmartsBuilder.atom(derived_mol.GetAtomWithIdx(atom_index))
        annotated_atoms = tuple(
            AnnotatedAtom(local_index=0, parameter=parameter)
            for parameter in AtomParameter
        )
        return Fragment(pattern=pattern, annotated_atoms=annotated_atoms)

    def _build_bond_fragment(
        self,
        derived_mol: Chem.Mol,
        atom_i: int,
        atom_j: int,
    ) -> Fragment:
        pattern = SmartsBuilder.chain(derived_mol, (atom_i, atom_j))
        annotated_bonds = tuple(
            AnnotatedBond(local_indices=(0, 1), parameter=parameter)
            for parameter in BondParameter
        )
        return Fragment(pattern=pattern, annotated_bonds=annotated_bonds)

    def _build_angle_fragment(
        self,
        derived_mol: Chem.Mol,
        atom_i: int,
        atom_j: int,
        atom_k: int,
    ) -> Fragment:
        pattern = SmartsBuilder.chain(derived_mol, (atom_i, atom_j, atom_k))
        annotated_angles = tuple(
            AnnotatedAngle(local_indices=(0, 1, 2), parameter=parameter)
            for parameter in AngleParameter
        )
        return Fragment(pattern=pattern, annotated_angles=annotated_angles)

    def _build_dihedral_fragment(
        self,
        derived_mol: Chem.Mol,
        atom_i: int,
        atom_j: int,
        atom_k: int,
        atom_l: int,
    ) -> Fragment:
        pattern = SmartsBuilder.chain(derived_mol, (atom_i, atom_j, atom_k, atom_l))
        annotated_dihedrals = tuple(
            AnnotatedDihedral(local_indices=(0, 1, 2, 3), parameter=parameter)
            for parameter in DihedralParameter
        )
        return Fragment(pattern=pattern, annotated_dihedrals=annotated_dihedrals)
