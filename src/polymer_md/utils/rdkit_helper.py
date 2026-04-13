from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol

from polymer_md.core import MolAtom


class RDKitHelper:
    @staticmethod
    def canonicalise(mol: Chem.Mol) -> Chem.Mol:
        ranks = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        new_order = sorted(range(mol.GetNumAtoms()), key=lambda i: ranks[i])
        return Chem.rdmolops.RenumberAtoms(mol, new_order)

    @staticmethod
    def replace_map_num(mol: Chem.Mol, old_map: int, new_map: int) -> Chem.Mol:
        rw = RWMol(mol)
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == old_map:
                a.SetAtomMapNum(new_map)
        return rw.GetMol()

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
    def single_bond_join_at_wildcard_sites(site1: MolAtom, site2: MolAtom) -> Chem.Mol:
        return RDKitHelper.join_many_at_map_nums(
            mol=site1.mol,
            mols_to_combine=[site2.mol],
            pairs=[(site1.atom.GetAtomMapNum(), site2.atom.GetAtomMapNum())],
        )

    @staticmethod
    def get_single_neighbour(mol_atom: MolAtom) -> MolAtom:
        neighbours = mol_atom.neighbours
        if len(neighbours) != 1:
            raise ValueError(
                f"Atom at index {mol_atom.idx} expected exactly 1 neighbour, "
                f"found {len(neighbours)}."
            )
        return neighbours[0]

    @staticmethod
    def relabel_wildcard(mol: Chem.Mol, new_map_num: int) -> Chem.Mol:
        star = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        rw = Chem.rdchem.RWMol(mol)
        rw.GetAtomWithIdx(star.GetIdx()).SetAtomMapNum(new_map_num)
        return rw.GetMol()

    @staticmethod
    def join_many_at_map_nums(
        mol: Chem.Mol,
        mols_to_combine: list[Chem.Mol],
        pairs: list[tuple[int, int]],
    ) -> Chem.Mol:
        combined = mol
        for m in mols_to_combine:
            combined = Chem.rdmolops.CombineMols(combined, m)

        rw = RWMol(combined)
        stars_to_remove = []

        for map1, map2 in pairs:
            s1 = RDKitHelper.get_site_idx(rw.GetMol(), atom_num=0, map_num=map1)
            s2 = RDKitHelper.get_site_idx(rw.GetMol(), atom_num=0, map_num=map2)
            a1 = RDKitHelper.get_single_neighbour(MolAtom(rw.GetMol(), s1)).idx
            a2 = RDKitHelper.get_single_neighbour(MolAtom(rw.GetMol(), s2)).idx
            rw.AddBond(a1, a2, Chem.rdchem.BondType.SINGLE)
            stars_to_remove.extend([s1, s2])

        for idx in sorted(set(stars_to_remove), reverse=True):
            rw.RemoveAtom(idx)

        Chem.rdmolops.SanitizeMol(rw)
        return rw.GetMol()

    @staticmethod
    def visualize_mol(
        mol: Chem.Mol, size=(300, 300), title: Optional[str] = None
    ) -> None:

        img = Draw.MolToImage(mol, size=size)
        plt.figure(figsize=(size[0] / 100, size[1] / 100))
        plt.imshow(img)
        if title is not None:
            plt.title(title)

        plt.axis("off")

        plt.show()
