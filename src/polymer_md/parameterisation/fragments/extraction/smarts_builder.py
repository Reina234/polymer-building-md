from __future__ import annotations

from rdkit import Chem


class SmartsBuilder:
    @staticmethod
    def atom(atom: Chem.Atom) -> str:
        atomic_number = atom.GetAtomicNum()
        aromaticity_flag = "a" if atom.GetIsAromatic() else "A"
        formal_charge = atom.GetFormalCharge()
        if formal_charge != 0:
            charge_sign = "+" if formal_charge > 0 else ""
            return f"[#{atomic_number};{aromaticity_flag};{charge_sign}{formal_charge}]"
        return f"[#{atomic_number};{aromaticity_flag}]"

    @staticmethod
    def bond(bond: Chem.Bond) -> str:
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            return "-"
        if bond_type == Chem.BondType.DOUBLE:
            return "="
        if bond_type == Chem.BondType.AROMATIC:
            return ":"
        if bond_type == Chem.BondType.TRIPLE:
            return "#"
        return "~"

    @staticmethod
    def chain(mol: Chem.Mol, atom_indices: tuple[int, ...]) -> str:
        parts = [SmartsBuilder.atom(mol.GetAtomWithIdx(atom_indices[0]))]
        for left_index, right_index in zip(atom_indices, atom_indices[1:]):
            bond = mol.GetBondBetweenAtoms(left_index, right_index)
            bond_smarts = SmartsBuilder.bond(bond) if bond is not None else "~"
            parts.append(bond_smarts)
            parts.append(SmartsBuilder.atom(mol.GetAtomWithIdx(right_index)))
        return "".join(parts)
