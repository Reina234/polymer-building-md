from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from polymer_md.core import MolAtom


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
    def remove_atom_and_adjust(rw: RWMol, atom_idx: int) -> None:
        rw.RemoveAtom(atom_idx)

    @staticmethod
    def anchor_idx_after_removal(anchor_idx: int, removed_idx: int) -> int:
        return anchor_idx if anchor_idx < removed_idx else anchor_idx - 1

    @staticmethod
    def combine_mols_at_indices(
        anchor1: MolAtom,
        anchor2: MolAtom,
        indices_to_remove: list[int],
        bond_type: Chem.rdchem.BondType = Chem.rdchem.BondType.SINGLE,
    ) -> Chem.Mol:
        combined = Chem.rdmolops.CombineMols(anchor1.mol, anchor2.mol)
        rw = RWMol(combined)

        if rw.GetBondBetweenAtoms(anchor1.idx, anchor2.idx) is not None:
            raise ValueError(
                f"Bond already exists between atoms {anchor1.idx} and {anchor2.idx}."
            )

        rw.AddBond(anchor1.idx, anchor2.idx, bond_type)

        for idx in sorted(indices_to_remove, reverse=True):
            rw.RemoveAtom(idx)

        Chem.rdmolops.SanitizeMol(rw)
        return rw.GetMol()

    @staticmethod
    def single_bond_join_at_wildcard_sites(
        site1: MolAtom,
        site2: MolAtom,
    ) -> Chem.Mol:
        offset = site1.mol.GetNumAtoms()
        adj_star_idx2 = site2.idx + offset

        combined = Chem.rdmolops.CombineMols(site1.mol, site2.mol)
        combined_site1 = MolAtom(mol=combined, idx=site1.idx)
        combined_site2 = MolAtom(mol=combined, idx=adj_star_idx2)

        assert (
            len(combined_site1.neighbours) == 1 and len(combined_site2.neighbours) == 1
        )
        anchor1 = combined_site1.neighbours[0]
        anchor2 = combined_site2.neighbours[0]

        return RDKitHelper.combine_mols_at_indices(
            anchor1=anchor1,
            anchor2=MolAtom(mol=anchor2.mol, idx=anchor2.idx - offset),
            indices_to_remove=[site1.idx, adj_star_idx2],
        )
