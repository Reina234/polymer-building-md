from __future__ import annotations

from dataclasses import dataclass

import parmed as pmd
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
)
from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder


@dataclass
class RegionFragmentExtractor:
    def extract(
        self,
        derived_mol: Chem.Mol,
        structure: pmd.Structure,
        region_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        return (
            self._extract_atom_fragments(derived_mol, region_parmed_indices)
            + self._extract_bond_fragments(derived_mol, structure, region_parmed_indices)
            + self._extract_angle_fragments(derived_mol, structure, region_parmed_indices)
            + self._extract_dihedral_fragments(derived_mol, structure, region_parmed_indices)
        )

    @staticmethod
    def resolve_parmed_indices(
        heavy_atom_indices: frozenset[int],
        mol_3d: Chem.Mol,
        mol3d_to_parmed: dict[int, int],
    ) -> frozenset[int]:
        hydrogen_indices = RegionFragmentExtractor._find_attached_hydrogen_indices(
            mol_3d, heavy_atom_indices
        )
        all_region_indices = heavy_atom_indices | hydrogen_indices
        return frozenset(
            mol3d_to_parmed[index]
            for index in all_region_indices
            if index in mol3d_to_parmed
        )

    @staticmethod
    def _find_attached_hydrogen_indices(
        mol_3d: Chem.Mol,
        heavy_atom_indices: frozenset[int],
    ) -> frozenset[int]:
        attached_hydrogens = set()
        for atom in mol_3d.GetAtoms():
            if atom.GetAtomicNum() != 1:
                continue
            parent_index = atom.GetNeighbors()[0].GetIdx()
            if parent_index in heavy_atom_indices:
                attached_hydrogens.add(atom.GetIdx())
        return frozenset(attached_hydrogens)

    def _extract_atom_fragments(
        self,
        derived_mol: Chem.Mol,
        region_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        return [
            self._build_atom_fragment(derived_mol, atom_index)
            for atom_index in region_parmed_indices
        ]

    def _extract_bond_fragments(
        self,
        derived_mol: Chem.Mol,
        structure: pmd.Structure,
        region_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        seen_keys: set[tuple[int, int]] = set()
        fragments = []
        for bond in structure.bonds:
            atom_i, atom_j = bond.atom1.idx, bond.atom2.idx
            if not self._touches_region(region_parmed_indices, atom_i, atom_j):
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
        region_parmed_indices: frozenset[int],
    ) -> list[Fragment]:
        seen_keys: set[tuple[int, int, int]] = set()
        fragments = []
        for angle in structure.angles:
            atom_i, atom_j, atom_k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
            if atom_j not in region_parmed_indices:
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
        region_parmed_indices: frozenset[int],
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
            if not self._middle_touches_region(region_parmed_indices, atom_j, atom_k):
                continue
            key = self._dihedral_key(atom_i, atom_j, atom_k, atom_l)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fragments.append(self._build_dihedral_fragment(derived_mol, key))
        return fragments

    def _touches_region(
        self,
        region: frozenset[int],
        atom_i: int,
        atom_j: int,
    ) -> bool:
        return atom_i in region or atom_j in region

    def _middle_touches_region(
        self,
        region: frozenset[int],
        atom_j: int,
        atom_k: int,
    ) -> bool:
        return atom_j in region or atom_k in region

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
        atom_indices: tuple[int, int, int, int],
    ) -> Fragment:
        pattern = SmartsBuilder.chain(derived_mol, atom_indices)
        annotated_dihedrals = tuple(
            AnnotatedDihedral(local_indices=(0, 1, 2, 3), parameter=parameter)
            for parameter in DihedralParameter
        )
        return Fragment(pattern=pattern, annotated_dihedrals=annotated_dihedrals)
