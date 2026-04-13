from __future__ import annotations

from typing import List

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from polymer_md.building.monomer import Monomer
from polymer_md.building.residue import AdditionPolymerResidue
from polymer_md.building.sites import PolymerisationSite
from polymer_md.utils.rdkit_helper import RDKitHelper


class MonomerToResidueConverter:

    def convert(self, monomer: Monomer) -> AdditionPolymerResidue:
        mol = RDKitHelper.mol_from_smiles(smiles=monomer.smiles)
        bonds = self._find_polymerisable_cc_bonds(mol=mol)
        self._validate_bond_is_polymerisable(bonds=bonds, monomer=monomer)
        residue_mol = self._open_double_bond(mol=mol, bond=bonds[0])
        canonical_mol = RDKitHelper.canonicalise(mol=residue_mol)
        residue_smiles = Chem.rdmolfiles.MolToSmiles(mol=canonical_mol)

        return AdditionPolymerResidue(
            residue_smiles=residue_smiles,
            label=monomer.label,
        )

    def _validate_bond_is_polymerisable(
        self, bonds: List[Chem.Bond], monomer: Monomer
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

    def _find_polymerisable_cc_bonds(self, mol: Chem.Mol) -> list[Chem.rdchem.Bond]:
        bonds = []
        for bond in mol.GetBonds():
            assert isinstance(bond, Chem.rdchem.Bond)
            if bond.GetBondTypeAsDouble() != 2.0:
                continue
            if bond.GetIsAromatic():
                continue
            a1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            a2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            if a1.GetSymbol() == "C" and a2.GetSymbol() == "C":
                bonds.append(bond)
        return bonds

    def _open_double_bond(
        self, mol: Chem.rdchem.Mol, bond: Chem.rdchem.Bond
    ) -> Chem.Mol:
        rw = RWMol(mol)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()

        rw.GetBondBetweenAtoms(idx1, idx2).SetBondType(Chem.rdchem.BondType.SINGLE)

        star1 = rw.AddAtom(self._make_wildcard(PolymerisationSite.HEAD))
        star2 = rw.AddAtom(self._make_wildcard(PolymerisationSite.TAIL))
        rw.AddBond(idx1, star1, Chem.rdchem.BondType.SINGLE)
        rw.AddBond(idx2, star2, Chem.rdchem.BondType.SINGLE)

        Chem.rdmolops.SanitizeMol(rw)
        return rw.GetMol()

    def _make_wildcard(self, site: PolymerisationSite) -> Chem.rdchem.Atom:
        atom = Chem.rdchem.Atom(0)
        atom.SetAtomMapNum(site)
        return atom
