from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder


# ---------------------------------------------------------------------------
# Tests: atom SMARTS
# ---------------------------------------------------------------------------

class TestSmartsBuilderAtom:
    def test_aliphatic_carbon(self):
        mol = Chem.MolFromSmiles("C")
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert result == "[#6;A]"

    def test_aromatic_carbon(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        Chem.SanitizeMol(mol)
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert result == "[#6;a]"

    def test_positive_formal_charge(self):
        mol = Chem.MolFromSmiles("[NH4+]")
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert "+1" in result or "+" in result
        assert "#7" in result

    def test_negative_formal_charge(self):
        mol = Chem.MolFromSmiles("[O-]")
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert "-1" in result
        assert "#8" in result

    def test_zero_charge_no_charge_in_smarts(self):
        mol = Chem.MolFromSmiles("CC")
        atom = mol.GetAtomWithIdx(0)
        result = SmartsBuilder.atom(atom)
        assert "+" not in result
        assert "-" not in result


# ---------------------------------------------------------------------------
# Tests: bond SMARTS
# ---------------------------------------------------------------------------

class TestSmartsBuilderBond:
    def test_single_bond(self):
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == "-"

    def test_double_bond(self):
        mol = Chem.MolFromSmiles("C=C")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == "="

    def test_aromatic_bond(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        Chem.SanitizeMol(mol)
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == ":"

    def test_triple_bond(self):
        mol = Chem.MolFromSmiles("C#C")
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert SmartsBuilder.bond(bond) == "#"

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
        mol = Chem.MolFromSmiles("CC")
        smarts, g2l = SmartsBuilder.subgraph(mol, ())
        assert smarts == ""
        assert g2l == {}

    def test_single_atom(self):
        mol = Chem.MolFromSmiles("C")
        smarts, g2l = SmartsBuilder.subgraph(mol, (0,))
        assert "#6" in smarts
        assert g2l == {0: 0}

    def test_two_atoms(self):
        mol = Chem.MolFromSmiles("CC")
        smarts, g2l = SmartsBuilder.subgraph(mol, (0, 1))
        assert len(g2l) == 2
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        assert mol.GetSubstructMatches(query)

    def test_linear_chain_matches_original(self):
        mol = Chem.MolFromSmiles("CCC")
        heavy = tuple(range(mol.GetNumAtoms()))
        smarts, g2l = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert mol.GetSubstructMatches(query)

    def test_global_to_local_contiguous(self):
        mol = Chem.MolFromSmiles("CCCC")
        heavy = (0, 1, 2, 3)
        _, g2l = SmartsBuilder.subgraph(mol, heavy)
        assert sorted(g2l.values()) == list(range(len(heavy)))

    def test_branched_smarts_matches(self):
        # Isobutane: branched C — exercises the (bond_smarts child) branch
        mol = Chem.MolFromSmiles("CC(C)C")
        heavy = tuple(range(mol.GetNumAtoms()))
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
        mol = Chem.MolFromSmiles("C1CCC1")  # cyclobutane
        heavy = tuple(range(mol.GetNumAtoms()))
        smarts, g2l = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert query is not None

    def test_ring_smarts_matches_mol(self):
        mol = Chem.MolFromSmiles("C1CCC1")
        heavy = tuple(range(mol.GetNumAtoms()))
        smarts, _ = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert mol.GetSubstructMatches(query)

    def test_benzene_smarts_matches(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        Chem.SanitizeMol(mol)
        heavy = tuple(range(mol.GetNumAtoms()))
        smarts, _ = SmartsBuilder.subgraph(mol, heavy)
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        assert mol.GetSubstructMatches(query)
