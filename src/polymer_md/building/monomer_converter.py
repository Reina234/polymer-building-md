from __future__ import annotations

from typing import List

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from polymer_md.building.data_models import AdditionPolymerResidue, MapLabels
from polymer_md.core import Monomer
from polymer_md.utils import RDKitHelper


class MonomerToResidueConverter:
    @classmethod
    def convert(cls, monomer: Monomer) -> AdditionPolymerResidue:
        mol = RDKitHelper.mol_from_smiles(smiles=monomer.smiles)
        bonds = cls._find_polymerisable_cc_bonds(mol=mol)
        cls._validate_bond_is_polymerisable(bonds=bonds, monomer=monomer)
        residue_mol = cls._open_double_bond(mol=mol, bond=bonds[0])
        canonical_mol = RDKitHelper.canonicalise(mol=residue_mol)
        residue_smiles = Chem.rdmolfiles.MolToSmiles(mol=canonical_mol)

        return AdditionPolymerResidue(
            residue_smiles=residue_smiles,
            label=monomer.label,
        )

    @classmethod
    def _validate_bond_is_polymerisable(
        cls, bonds: List[Chem.Bond], monomer: Monomer
    ) -> None:
        if len(bonds) == 0:
            raise ValueError(
                f"No polymerisable non-aromatic C=C bond found in {monomer.smiles!r}."
            )
        if len(bonds) > 1:
            raise ValueError(
                f"Found {len(bonds)} non-aromatic C=C bonds in {monomer.smiles!r}. "
                "Exactly one required for addition polymerisation."
            )

    @classmethod
    def _find_polymerisable_cc_bonds(cls, mol: Chem.Mol) -> list[Chem.rdchem.Bond]:
        bonds = []
        for bond in mol.GetBonds():
            assert isinstance(bond, Chem.rdchem.Bond)
            if not cls._could_be_polymerisable(bond=bond):
                continue
            a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            if a1.GetSymbol() == "C" and a2.GetSymbol() == "C":
                bonds.append(bond)
        return bonds

    @classmethod
    def _could_be_polymerisable(cls, bond: Chem.rdchem.Bond) -> bool:
        if bond.GetBondTypeAsDouble() != 2.0:
            return False
        if bond.GetIsAromatic():
            return False
        return True

    @classmethod
    def _open_double_bond(
        cls, mol: Chem.rdchem.Mol, bond: Chem.rdchem.Bond
    ) -> Chem.Mol:
        head_idx, tail_idx = cls._assign_head_tail(mol, bond)
        rw = RWMol(mol)

        rw.GetBondBetweenAtoms(head_idx, tail_idx).SetBondType(
            Chem.rdchem.BondType.SINGLE
        )

        star_head = rw.AddAtom(cls._make_wildcard(MapLabels.HEAD))
        star_tail = rw.AddAtom(cls._make_wildcard(MapLabels.TAIL))
        rw.AddBond(head_idx, star_head, Chem.rdchem.BondType.SINGLE)
        rw.AddBond(tail_idx, star_tail, Chem.rdchem.BondType.SINGLE)

        Chem.rdmolops.SanitizeMol(rw)
        return rw.GetMol()

    @classmethod
    def _assign_head_tail(
        cls, mol: Chem.rdchem.Mol, bond: Chem.rdchem.Bond
    ) -> tuple[int, int]:
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        sub1 = cls._heavy_substituent_count(mol, idx1, exclude=idx2)
        sub2 = cls._heavy_substituent_count(mol, idx2, exclude=idx1)
        if sub1 >= sub2:
            return idx1, idx2
        return idx2, idx1

    @staticmethod
    def _heavy_substituent_count(
        mol: Chem.rdchem.Mol, atom_idx: int, exclude: int
    ) -> int:
        atom = mol.GetAtomWithIdx(atom_idx)
        return sum(
            1
            for nb in atom.GetNeighbors()
            if nb.GetIdx() != exclude and nb.GetAtomicNum() != 1
        )

    @classmethod
    def _make_wildcard(cls, site: MapLabels) -> Chem.rdchem.Atom:
        atom = Chem.rdchem.Atom(0)
        atom.SetAtomMapNum(site)
        return atom
