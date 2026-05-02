from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder


# ---------------------------------------------------------------------------
# Tests: atom SMARTS
# ---------------------------------------------------------------------------

class TestSmartsBuilderAtom:
    def test_aliphatic_carbon_h_count_included(self):
        # Builder is used on explicit-H molecules (removeHs=False from DetermineBonds).
        # H-count constraint distinguishes CH3/CH2/CH/quaternary without GAFF2 types.
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        atom = mol.GetAtomWithIdx(0)  # methane carbon, 4 explicit H neighbours
        result = SmartsBuilder.atom(atom)
        assert result == "[#6;A;H4]"

    def test_aromatic_carbon_h_count_included(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        Chem.SanitizeMol(mol)
        atom = mol.GetAtomWithIdx(0)  # benzene carbon, 1 explicit H
        result = SmartsBuilder.atom(atom)
        assert result == "[#6;a;H1]"

    def test_ch2_carbon(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        atom = mol.GetAtomWithIdx(0)  # ethane terminal carbon, 3 explicit H
        result = SmartsBuilder.atom(atom)
        assert result == "[#6;A;H3]"

    def test_quaternary_carbon_h_count_zero(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)C"))
        atom = mol.GetAtomWithIdx(1)  # central quaternary carbon, 0 H
        result = SmartsBuilder.atom(atom)
        assert result == "[#6;A;H0]"

    def test_hydrogen_atom_no_h_part(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        atom = mol.GetAtomWithIdx(1)  # an explicit H atom
        result = SmartsBuilder.atom(atom)
        assert result == "[#1;A]"  # no ;H on H atoms

    def test_positive_formal_charge(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert "+" in result
        assert "#7" in result

    def test_negative_formal_charge(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[O-]"))
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert "-" in result
        assert "#8" in result

    def test_zero_charge_no_charge_in_smarts(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert "+" not in result
        assert "-" not in result


# ---------------------------------------------------------------------------
# Tests: bond SMARTS
# ---------------------------------------------------------------------------

class TestSmartsBuilderBond:
    def test_single_bond_returns_tilde(self):
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == "~"

    def test_double_bond_returns_tilde(self):
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == "~"

    def test_aromatic_bond(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        Chem.SanitizeMol(mol)
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == ":"

    def test_triple_bond_returns_tilde(self):
        mol = Chem.MolFromSmiles("C#C")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == "~"

    def test_unknown_bond_type_returns_tilde(self):
        from unittest.mock import MagicMock
        bond = MagicMock()
        bond.GetBondType.return_value = 9999
        assert SmartsBuilder.bond(bond) == "~"


# ---------------------------------------------------------------------------
# Tests: subgraph — linear chains
# ---------------------------------------------------------------------------

class TestSubgraphLinear:
    def test_empty_returns_empty(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        smarts, g2l = SmartsBuilder.subgraph(mol, ())
        assert smarts == ""
        assert g2l == {}

    def test_single_atom(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        smarts, g2l = SmartsBuilder.subgraph(mol, (0,))
        assert "#6" in smarts
        assert g2l == {0: 0}

    def test_two_atoms(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        smarts, g2l = SmartsBuilder.subgraph(mol, (0, 1))
        assert len(g2l) == 2
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        assert mol.GetSubstructMatches(query)

    def test_linear_chain_matches_original(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))
        heavy = tuple(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        smarts, g2l = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert mol.GetSubstructMatches(query)

    def test_global_to_local_contiguous(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
        heavy = tuple(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        _, g2l = SmartsBuilder.subgraph(mol, heavy)
        assert sorted(g2l.values()) == list(range(len(heavy)))

    def test_branched_smarts_matches(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)C"))
        heavy = tuple(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        smarts, g2l = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        matches = mol.GetSubstructMatches(query)
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# Tests: subgraph — rings (back-edge closure)
# ---------------------------------------------------------------------------

class TestSubgraphRing:
    def test_ring_smarts_is_valid(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C1CCC1"))
        heavy = tuple(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        smarts, g2l = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert query is not None

    def test_ring_smarts_matches_mol(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C1CCC1"))
        heavy = tuple(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        smarts, _ = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert mol.GetSubstructMatches(query)

    def test_benzene_smarts_matches(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        Chem.SanitizeMol(mol)
        heavy = tuple(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        smarts, _ = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        assert mol.GetSubstructMatches(query)
