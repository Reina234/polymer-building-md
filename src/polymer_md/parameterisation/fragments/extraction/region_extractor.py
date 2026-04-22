from __future__ import annotations

from dataclasses import dataclass

import parmed as pmd
from rdkit import Chem

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
    AnnotatedMember,
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
        context_parmed_indices: frozenset[int],
        region_parmed_indices: frozenset[int],
    ) -> tuple[Fragment, dict[int, int]]:
        pattern, global_to_local = SmartsBuilder.subgraph(
            derived_mol, tuple(sorted(context_parmed_indices))
        )
        members: list[AnnotatedMember] = (
            self._extract_atom_members(region_parmed_indices, global_to_local)
            + self._extract_bond_members(structure, region_parmed_indices, global_to_local)
            + self._extract_angle_members(structure, region_parmed_indices, global_to_local)
            + self._extract_dihedral_members(structure, region_parmed_indices, global_to_local)
        )
        fragment = Fragment(
            pattern=pattern,
            annotated_atoms=tuple(m for m in members if isinstance(m, AnnotatedAtom)),
            annotated_bonds=tuple(m for m in members if isinstance(m, AnnotatedBond)),
            annotated_angles=tuple(m for m in members if isinstance(m, AnnotatedAngle)),
            annotated_dihedrals=tuple(m for m in members if isinstance(m, AnnotatedDihedral)),
        )
        return fragment, global_to_local

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

    def _extract_atom_members(
        self,
        region_parmed_indices: frozenset[int],
        global_to_local: dict[int, int],
    ) -> list[AnnotatedAtom]:
        members = []
        for atom_idx in sorted(region_parmed_indices):
            if atom_idx not in global_to_local:
                continue
            local_idx = global_to_local[atom_idx]
            for parameter in AtomParameter:
                members.append(AnnotatedAtom(local_index=local_idx, parameter=parameter))
        return members

    def _extract_bond_members(
        self,
        structure: pmd.Structure,
        region_parmed_indices: frozenset[int],
        global_to_local: dict[int, int],
    ) -> list[AnnotatedBond]:
        seen: set[tuple[int, int]] = set()
        members = []
        for bond in structure.bonds:
            atom_i, atom_j = bond.atom1.idx, bond.atom2.idx
            if not self._touches_region(region_parmed_indices, atom_i, atom_j):
                continue
            if not self._all_in_context(global_to_local, atom_i, atom_j):
                continue
            key = self._sorted_pair(atom_i, atom_j)
            if key in seen:
                continue
            seen.add(key)
            local_i, local_j = global_to_local[atom_i], global_to_local[atom_j]
            local_key = self._sorted_pair(local_i, local_j)
            for parameter in BondParameter:
                members.append(AnnotatedBond(local_indices=local_key, parameter=parameter))
        return members

    def _extract_angle_members(
        self,
        structure: pmd.Structure,
        region_parmed_indices: frozenset[int],
        global_to_local: dict[int, int],
    ) -> list[AnnotatedAngle]:
        seen: set[tuple[int, int, int]] = set()
        members = []
        for angle in structure.angles:
            atom_i, atom_j, atom_k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
            if atom_j not in region_parmed_indices:
                continue
            if not self._all_in_context(global_to_local, atom_i, atom_j, atom_k):
                continue
            key = (min(atom_i, atom_k), atom_j, max(atom_i, atom_k))
            if key in seen:
                continue
            seen.add(key)
            local_i = global_to_local[atom_i]
            local_j = global_to_local[atom_j]
            local_k = global_to_local[atom_k]
            local_key = (local_i, local_j, local_k)
            for parameter in AngleParameter:
                members.append(AnnotatedAngle(local_indices=local_key, parameter=parameter))
        return members

    def _extract_dihedral_members(
        self,
        structure: pmd.Structure,
        region_parmed_indices: frozenset[int],
        global_to_local: dict[int, int],
    ) -> list[AnnotatedDihedral]:
        seen: set[tuple[int, int, int, int]] = set()
        members = []
        for dihedral in structure.dihedrals:
            if dihedral.improper:
                continue
            atom_i = dihedral.atom1.idx
            atom_j = dihedral.atom2.idx
            atom_k = dihedral.atom3.idx
            atom_l = dihedral.atom4.idx
            if not self._touches_region(region_parmed_indices, atom_j, atom_k):
                continue
            if not self._all_in_context(global_to_local, atom_i, atom_j, atom_k, atom_l):
                continue
            forward = (atom_i, atom_j, atom_k, atom_l)
            reverse = (atom_l, atom_k, atom_j, atom_i)
            key = min(forward, reverse)
            if key in seen:
                continue
            seen.add(key)
            local_key = tuple(global_to_local[idx] for idx in key)
            for parameter in DihedralParameter:
                members.append(AnnotatedDihedral(local_indices=local_key, parameter=parameter))
        return members

    def _touches_region(
        self,
        region: frozenset[int],
        atom_i: int,
        atom_j: int,
    ) -> bool:
        return atom_i in region or atom_j in region

    @staticmethod
    def _all_in_context(global_to_local: dict[int, int], *indices: int) -> bool:
        return all(idx in global_to_local for idx in indices)

    def _sorted_pair(self, atom_i: int, atom_j: int) -> tuple[int, int]:
        return (min(atom_i, atom_j), max(atom_i, atom_j))
