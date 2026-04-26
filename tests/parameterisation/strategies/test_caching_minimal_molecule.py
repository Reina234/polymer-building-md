from __future__ import annotations

from unittest.mock import MagicMock, patch

import parmed as pmd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.data_models.parameters import BondParameter, DihedralParameter, DihedralTerm
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.strategies.base import StrategyContext
from polymer_md.parameterisation.strategies.caching_minimal_molecule import CachingMinimalMoleculeStrategy
from polymer_md.parameterisation.strategies.minimal_molecule import MinimalMoleculeStrategy


def _make_propane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


def _empty_context(mol: Chem.Mol) -> StrategyContext:
    return StrategyContext(
        library=FragmentLibrary(records=(), atom_metadata={}),
        derived_mol=mol,
        polymer_atom_metadata={},
    )


def _make_structure_with_bond(i: int, j: int, k: float = 300.0, req: float = 1.54) -> pmd.Structure:
    structure = pmd.Structure()
    for _ in range(max(i, j) + 1):
        structure.add_atom(pmd.Atom(), "MOL", 1)
    bond = pmd.Bond(list(structure.atoms)[i], list(structure.atoms)[j])
    bond.type = pmd.BondType(k=k, req=req)
    structure.bonds.append(bond)
    return structure


class TestCachingMinimalMoleculeStrategyCallsInnerOnce:
    def test_inner_resolve_called_once_for_same_indices(self):
        mol = _make_propane_mol()
        context = _empty_context(mol)

        mock_inner = MagicMock(spec=MinimalMoleculeStrategy)
        mock_inner.expander = MagicMock()
        mock_inner.expander.expand.return_value = frozenset(range(mol.GetNumAtoms()))
        mock_inner.resolve.return_value = 300.0

        strategy = CachingMinimalMoleculeStrategy(inner=mock_inner)
        strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)
        strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)

        assert mock_inner.resolve.call_count == 1

    def test_inner_resolve_called_for_different_indices(self):
        mol = _make_propane_mol()
        context = _empty_context(mol)

        mock_inner = MagicMock(spec=MinimalMoleculeStrategy)
        mock_inner.expander = MagicMock()
        mock_inner.expander.expand.side_effect = lambda m, seeds: frozenset(range(mol.GetNumAtoms()))
        mock_inner.resolve.return_value = 300.0

        strategy = CachingMinimalMoleculeStrategy(inner=mock_inner)
        strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)
        strategy.resolve((1, 2), BondParameter.FORCE_CONSTANT, context)

        assert mock_inner.resolve.call_count == 2

    def test_cached_value_returned_on_second_call(self):
        mol = _make_propane_mol()
        context = _empty_context(mol)

        mock_inner = MagicMock(spec=MinimalMoleculeStrategy)
        mock_inner.expander = MagicMock()
        mock_inner.expander.expand.return_value = frozenset(range(mol.GetNumAtoms()))
        mock_inner.resolve.return_value = 42.0

        strategy = CachingMinimalMoleculeStrategy(inner=mock_inner)
        first = strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)
        second = strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)

        assert first == 42.0
        assert second == 42.0

    def test_different_parameters_use_separate_cache_entries(self):
        mol = _make_propane_mol()
        context = _empty_context(mol)

        mock_inner = MagicMock(spec=MinimalMoleculeStrategy)
        mock_inner.expander = MagicMock()
        mock_inner.expander.expand.return_value = frozenset(range(mol.GetNumAtoms()))
        mock_inner.resolve.side_effect = [1.0, 2.0]

        strategy = CachingMinimalMoleculeStrategy(inner=mock_inner)
        r1 = strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)
        r2 = strategy.resolve((0, 1), BondParameter.EQUILIBRIUM_LENGTH, context)

        assert r1 == 1.0
        assert r2 == 2.0
        assert mock_inner.resolve.call_count == 2


class TestBuildKey:
    def test_key_is_string(self):
        mol = _make_propane_mol()
        key = CachingMinimalMoleculeStrategy._build_key(mol, (0, 1), BondParameter.FORCE_CONSTANT)
        assert isinstance(key, str)

    def test_same_mol_same_indices_same_key(self):
        mol = _make_propane_mol()
        k1 = CachingMinimalMoleculeStrategy._build_key(mol, (0, 1), BondParameter.FORCE_CONSTANT)
        k2 = CachingMinimalMoleculeStrategy._build_key(mol, (0, 1), BondParameter.FORCE_CONSTANT)
        assert k1 == k2

    def test_different_parameter_different_key(self):
        mol = _make_propane_mol()
        k1 = CachingMinimalMoleculeStrategy._build_key(mol, (0, 1), BondParameter.FORCE_CONSTANT)
        k2 = CachingMinimalMoleculeStrategy._build_key(mol, (0, 1), BondParameter.EQUILIBRIUM_LENGTH)
        assert k1 != k2
