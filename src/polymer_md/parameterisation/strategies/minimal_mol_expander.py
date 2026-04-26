from __future__ import annotations

from typing import Protocol

from rdkit import Chem


class MinimalMoleculeExpander(Protocol):
    def expand(self, mol: Chem.Mol, seed_indices: frozenset[int]) -> frozenset[int]: ...


class GAFFMinimalMoleculeExpander:
    def expand(self, mol: Chem.Mol, seed_indices: frozenset[int]) -> frozenset[int]:
        atom_set = set(seed_indices)
        for idx in seed_indices:
            self._include_atom_context(mol, idx, atom_set)
        self._close_fused_rings(mol, atom_set)
        self._add_valence_cap_neighbors(mol, atom_set)
        return frozenset(atom_set)

    @staticmethod
    def _close_fused_rings(mol: Chem.Mol, atom_set: set[int]) -> None:
        prev_size = -1
        while len(atom_set) != prev_size:
            prev_size = len(atom_set)
            GAFFMinimalMoleculeExpander._expand_ring_pass(mol, atom_set)

    @staticmethod
    def _expand_ring_pass(mol: Chem.Mol, atom_set: set[int]) -> None:
        for idx in list(atom_set):
            if mol.GetAtomWithIdx(idx).IsInRing():
                GAFFMinimalMoleculeExpander._add_containing_rings(mol, idx, atom_set)

    @staticmethod
    def _include_atom_context(mol: Chem.Mol, idx: int, atom_set: set[int]) -> None:
        if mol.GetAtomWithIdx(idx).IsInRing():
            GAFFMinimalMoleculeExpander._add_containing_rings(mol, idx, atom_set)
        else:
            GAFFMinimalMoleculeExpander._add_neighbors(mol, idx, atom_set)

    @staticmethod
    def _add_containing_rings(mol: Chem.Mol, idx: int, atom_set: set[int]) -> None:
        for ring in mol.GetRingInfo().AtomRings():
            if idx in ring:
                atom_set.update(ring)

    @staticmethod
    def _add_neighbors(mol: Chem.Mol, idx: int, atom_set: set[int]) -> None:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            atom_set.add(neighbor.GetIdx())

    @staticmethod
    def _add_valence_cap_neighbors(mol: Chem.Mol, atom_set: set[int]) -> None:
        for idx in list(atom_set):
            GAFFMinimalMoleculeExpander._add_neighbors(mol, idx, atom_set)
