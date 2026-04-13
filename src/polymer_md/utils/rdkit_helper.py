from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol


class RDKitHelper:
    @staticmethod
    def canonicalise(mol: Chem.Mol) -> Chem.Mol:
        ranks = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        new_order = sorted(range(mol.GetNumAtoms()), key=lambda i: ranks[i])
        return Chem.rdmolops.RenumberAtoms(mol, new_order)

    @staticmethod
    def get_site_idx(mol: Chem.Mol, atom_num: int, map_num: int) -> int:

        return next(
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == atom_num and a.GetAtomMapNum() == map_num
        )

    @staticmethod
    def mol_from_smiles(smiles: str) -> Chem.Mol:
        mol = Chem.rdmolfiles.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        return mol

    @staticmethod
    def replace_with_placeholder(
        mol: Chem.Mol, index_to_replace: int, placeholder_map_num: int
    ) -> Chem.Mol:
        rw = RWMol(mol)
        rw.GetAtomWithIdx(index_to_replace).SetAtomMapNum(placeholder_map_num)
        Chem.rdmolops.SanitizeMol(rw)
        return rw.GetMol()

    @staticmethod
    def get_single_neighbour_idx(mol: Chem.Mol, atom_idx: int) -> int:
        neighbours = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if len(neighbours) != 1:
            raise ValueError(
                f"Atom at index {atom_idx} expected exactly 1 neighbour, "
                f"found {len(neighbours)}."
            )
        return neighbours[0].GetIdx()

    @staticmethod
    def remove_atom_and_adjust(rw: RWMol, atom_idx: int) -> None:
        rw.RemoveAtom(atom_idx)

    @staticmethod
    def anchor_idx_after_removal(anchor_idx: int, removed_idx: int) -> int:
        return anchor_idx if anchor_idx < removed_idx else anchor_idx - 1

    @staticmethod
    def combine_mols_at_indices(
        mol1: Chem.Mol,
        anchor_idx1: int,
        mol2: Chem.Mol,
        anchor_idx2: int,
        indices_to_remove: list[int],
        bond_type: Chem.rdchem.BondType = Chem.rdchem.BondType.SINGLE,
    ) -> Chem.Mol:
        combined = Chem.rdmolops.CombineMols(mol1, mol2)
        rw = RWMol(combined)

        if rw.GetBondBetweenAtoms(anchor_idx1, anchor_idx2) is not None:
            raise ValueError(
                f"Bond already exists between atoms {anchor_idx1} and {anchor_idx2}."
            )

        rw.AddBond(anchor_idx1, anchor_idx2, bond_type)

        for idx in sorted(indices_to_remove, reverse=True):
            rw.RemoveAtom(idx)

        Chem.rdmolops.SanitizeMol(rw)
        return rw.GetMol()

    @staticmethod
    def single_bond_join_at_wildcard_sites(
        mol1: Chem.Mol,
        star_idx1: int,
        mol2: Chem.Mol,
        star_idx2: int,
    ) -> Chem.Mol:
        offset = mol1.GetNumAtoms()
        adj_star_idx2 = star_idx2 + offset

        combined = Chem.rdmolops.CombineMols(mol1, mol2)

        anchor_idx1 = RDKitHelper.get_single_neighbour_idx(
            mol=combined, atom_idx=star_idx1
        )
        anchor_idx2 = RDKitHelper.get_single_neighbour_idx(
            mol=combined, atom_idx=adj_star_idx2
        )

        return RDKitHelper.combine_mols_at_indices(
            mol1=mol1,
            anchor_idx1=anchor_idx1,
            mol2=mol2,
            anchor_idx2=anchor_idx2 - offset,
            indices_to_remove=[star_idx1, adj_star_idx2],
        )
