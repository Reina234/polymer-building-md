from __future__ import annotations

from rdkit import Chem


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
