from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import parmed as pmd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
)
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext
from polymer_md.parameterisation.strategies.minimal_molecule import MinimalMoleculeStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def butane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def propane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


def _empty_library() -> FragmentLibrary:
    return FragmentLibrary(records=(), atom_metadata={})


def _minimal_context(mol: Chem.Mol) -> StrategyContext:
    return StrategyContext(
        library=_empty_library(),
        derived_mol=mol,
        polymer_atom_metadata={},
    )


def _make_structure_with_bond(i: int, j: int, k: float, req: float) -> pmd.Structure:
    structure = pmd.Structure()
    for _ in range(max(i, j) + 1):
        structure.add_atom(pmd.Atom(), "MOL", 1)
    atoms = list(structure.atoms)
    bond = pmd.Bond(atoms[i], atoms[j])
    bond.type = pmd.BondType(k=k, req=req)
    structure.bonds.append(bond)
    return structure


def _make_structure_with_angle(
    i: int, j: int, k: int, force_k: float, theteq: float
) -> pmd.Structure:
    structure = pmd.Structure()
    for _ in range(max(i, j, k) + 1):
        structure.add_atom(pmd.Atom(), "MOL", 1)
    atoms = list(structure.atoms)
    angle = pmd.Angle(atoms[i], atoms[j], atoms[k])
    angle.type = pmd.AngleType(k=force_k, theteq=theteq)
    structure.angles.append(angle)
    return structure


def _make_structure_with_dihedral(
    i: int, j: int, k: int, l: int,
    phi_k: float, phase: float, per: float,
    improper: bool = False,
) -> pmd.Structure:
    structure = pmd.Structure()
    for _ in range(max(i, j, k, l) + 1):
        structure.add_atom(pmd.Atom(), "MOL", 1)
    atoms = list(structure.atoms)
    dihedral = pmd.Dihedral(atoms[i], atoms[j], atoms[k], atoms[l], improper=improper)
    dihedral.type = pmd.DihedralType(phi_k=phi_k, phase=phase, per=per)
    structure.dihedrals.append(dihedral)
    return structure


# ---------------------------------------------------------------------------
# Tests: raises for AtomParameter
# ---------------------------------------------------------------------------

class TestMinimalMoleculeStrategyRaisesForAtomParameter:
    def test_raises_for_charge(self, propane_mol):
        strategy = MinimalMoleculeStrategy()
        context = _minimal_context(propane_mol)
        with pytest.raises(MissingParameterError, match="AtomParameter"):
            strategy.resolve((0,), AtomParameter.CHARGE, context)

    def test_raises_for_epsilon(self, propane_mol):
        strategy = MinimalMoleculeStrategy()
        context = _minimal_context(propane_mol)
        with pytest.raises(MissingParameterError, match="AtomParameter"):
            strategy.resolve((0,), AtomParameter.EPSILON, context)


# ---------------------------------------------------------------------------
# Tests: _extract_minimal_mol
# ---------------------------------------------------------------------------

class TestExtractMinimalMol:
    def test_includes_core_atoms_and_neighbours(self, butane_mol):
        # butane heavy atoms: 0, 1, 2, 3 (C-C-C-C)
        # For bond (1, 2), core atoms are 1 and 2; neighbours add 0 and 3
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(butane_mol, (1, 2))

        assert 1 in old_to_new
        assert 2 in old_to_new
        # Neighbours of 1 (C at index 1): C at 0 and C at 2 and some H
        # Neighbours of 2 (C at index 2): C at 1 and C at 3 and some H
        assert len(old_to_new) >= 4  # at minimum: 0, 1, 2, 3

    def test_old_to_new_maps_core_atoms(self, propane_mol):
        # heavy atom indices: 0, 1, 2
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(propane_mol, (0, 1))
        assert 0 in old_to_new
        assert 1 in old_to_new

    def test_output_mol_has_explicit_h(self, propane_mol):
        mol, _ = MinimalMoleculeStrategy._extract_minimal_mol(propane_mol, (0, 1))
        h_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
        assert h_count > 0

    def test_local_indices_contiguous_from_zero(self, butane_mol):
        _, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(butane_mol, (1, 2))
        new_indices = sorted(old_to_new.values())
        assert new_indices == list(range(len(new_indices)))

    def test_bonds_preserved_within_extracted_set(self, butane_mol):
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(butane_mol, (1, 2))
        bond_pairs = {
            frozenset({b.GetBeginAtomIdx(), b.GetEndAtomIdx()})
            for b in mol.GetBonds()
        }
        new_1 = old_to_new[1]
        new_2 = old_to_new[2]
        assert frozenset({new_1, new_2}) in bond_pairs

    def test_single_atom_extracts_with_neighbours(self, propane_mol):
        # For a single atom, extract that atom + all its neighbours
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(propane_mol, (1,))
        assert 1 in old_to_new
        assert len(old_to_new) > 1  # at least the middle C and its neighbours


# ---------------------------------------------------------------------------
# Tests: _bond_value
# ---------------------------------------------------------------------------

class TestBondValue:
    def test_returns_force_constant(self):
        structure = _make_structure_with_bond(0, 1, k=300.0, req=1.54)
        result = MinimalMoleculeStrategy._bond_value(structure, (0, 1), BondParameter.FORCE_CONSTANT)
        assert pytest.approx(result, abs=1e-6) == 300.0

    def test_returns_equilibrium_length(self):
        structure = _make_structure_with_bond(0, 1, k=300.0, req=1.54)
        result = MinimalMoleculeStrategy._bond_value(structure, (0, 1), BondParameter.EQUILIBRIUM_LENGTH)
        assert pytest.approx(result, abs=1e-6) == 1.54

    def test_order_independent(self):
        structure = _make_structure_with_bond(0, 1, k=300.0, req=1.54)
        fwd = MinimalMoleculeStrategy._bond_value(structure, (0, 1), BondParameter.FORCE_CONSTANT)
        rev = MinimalMoleculeStrategy._bond_value(structure, (1, 0), BondParameter.FORCE_CONSTANT)
        assert fwd == rev

    def test_raises_when_bond_not_found(self):
        structure = _make_structure_with_bond(0, 1, k=300.0, req=1.54)
        with pytest.raises(MissingParameterError, match="Bond not found"):
            MinimalMoleculeStrategy._bond_value(structure, (0, 2), BondParameter.FORCE_CONSTANT)


# ---------------------------------------------------------------------------
# Tests: _angle_value
# ---------------------------------------------------------------------------

class TestAngleValue:
    def test_returns_force_constant(self):
        structure = _make_structure_with_angle(0, 1, 2, force_k=50.0, theteq=109.5)
        result = MinimalMoleculeStrategy._angle_value(structure, (0, 1, 2), AngleParameter.FORCE_CONSTANT)
        assert pytest.approx(result, abs=1e-6) == 50.0

    def test_returns_equilibrium_angle(self):
        structure = _make_structure_with_angle(0, 1, 2, force_k=50.0, theteq=109.5)
        result = MinimalMoleculeStrategy._angle_value(structure, (0, 1, 2), AngleParameter.EQUILIBRIUM_ANGLE)
        assert pytest.approx(result, abs=1e-6) == 109.5

    def test_raises_when_angle_not_found(self):
        structure = _make_structure_with_angle(0, 1, 2, force_k=50.0, theteq=109.5)
        with pytest.raises(MissingParameterError, match="Angle not found"):
            MinimalMoleculeStrategy._angle_value(structure, (0, 1, 3), AngleParameter.FORCE_CONSTANT)


# ---------------------------------------------------------------------------
# Tests: _dihedral_value
# ---------------------------------------------------------------------------

class TestDihedralValue:
    def test_returns_force_constant(self):
        structure = _make_structure_with_dihedral(0, 1, 2, 3, phi_k=0.5, phase=0.0, per=3)
        result = MinimalMoleculeStrategy._dihedral_value(
            structure, (0, 1, 2, 3), DihedralParameter.FORCE_CONSTANT
        )
        assert pytest.approx(result, abs=1e-6) == 0.5

    def test_returns_phase(self):
        structure = _make_structure_with_dihedral(0, 1, 2, 3, phi_k=0.5, phase=1.57, per=3)
        result = MinimalMoleculeStrategy._dihedral_value(
            structure, (0, 1, 2, 3), DihedralParameter.PHASE
        )
        assert pytest.approx(result, abs=1e-4) == 1.57

    def test_returns_periodicity(self):
        structure = _make_structure_with_dihedral(0, 1, 2, 3, phi_k=0.5, phase=0.0, per=2)
        result = MinimalMoleculeStrategy._dihedral_value(
            structure, (0, 1, 2, 3), DihedralParameter.PERIODICITY
        )
        assert pytest.approx(result, abs=1e-6) == 2.0

    def test_skips_improper_dihedrals(self):
        # Improper dihedral should be skipped; no proper dihedral → raises
        structure = _make_structure_with_dihedral(0, 1, 2, 3, phi_k=0.5, phase=0.0, per=3, improper=True)
        with pytest.raises(MissingParameterError, match="Dihedral not found"):
            MinimalMoleculeStrategy._dihedral_value(
                structure, (0, 1, 2, 3), DihedralParameter.FORCE_CONSTANT
            )

    def test_raises_when_dihedral_not_found(self):
        structure = _make_structure_with_dihedral(0, 1, 2, 3, phi_k=0.5, phase=0.0, per=3)
        with pytest.raises(MissingParameterError, match="Dihedral not found"):
            MinimalMoleculeStrategy._dihedral_value(
                structure, (0, 1, 2, 9), DihedralParameter.FORCE_CONSTANT
            )


# ---------------------------------------------------------------------------
# Tests: _read_parameter dispatch
# ---------------------------------------------------------------------------

class TestReadParameter:
    def test_dispatches_to_bond(self):
        structure = _make_structure_with_bond(0, 1, k=300.0, req=1.54)
        result = MinimalMoleculeStrategy._read_parameter(
            structure, (0, 1), BondParameter.FORCE_CONSTANT
        )
        assert pytest.approx(result, abs=1e-6) == 300.0

    def test_dispatches_to_angle(self):
        structure = _make_structure_with_angle(0, 1, 2, force_k=50.0, theteq=109.5)
        result = MinimalMoleculeStrategy._read_parameter(
            structure, (0, 1, 2), AngleParameter.FORCE_CONSTANT
        )
        assert pytest.approx(result, abs=1e-6) == 50.0

    def test_dispatches_to_dihedral(self):
        structure = _make_structure_with_dihedral(0, 1, 2, 3, phi_k=1.0, phase=0.0, per=3)
        result = MinimalMoleculeStrategy._read_parameter(
            structure, (0, 1, 2, 3), DihedralParameter.FORCE_CONSTANT
        )
        assert pytest.approx(result, abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# Tests: resolve (mocked _parameterise)
# ---------------------------------------------------------------------------

class TestResolveWithMockedParameterisation:
    def test_resolve_bond_parameter(self, propane_mol):
        structure = _make_structure_with_bond(0, 1, k=300.0, req=1.54)

        strategy = MinimalMoleculeStrategy()
        context = _minimal_context(propane_mol)

        with patch.object(strategy, "_parameterise", return_value=structure):
            # heavy atoms in propane: 0=C, 1=C, 2=C
            # For bond (0, 1): extract 0, 1, their neighbours → indices map 0→0, 1→1 (plus H)
            # local_indices = (old_to_new[0], old_to_new[1]) = (0, 1) if 0,1 are first sorted
            result = strategy.resolve((0, 1), BondParameter.FORCE_CONSTANT, context)

        assert pytest.approx(result, abs=1e-6) == 300.0

    def test_resolve_angle_parameter(self, propane_mol):
        # In propane (CCC + H): angle 0-1-2
        # Extract atoms (0,1,2) + neighbours
        # old_to_new: depends on sorted(atom_set)
        # atom_set for angle (0,1,2): {0,1,2} + neighbours of 0,1,2
        # neighbours of 0: [1, H...]; neighbours of 1: [0, 2, H...]; neighbours of 2: [1, H...]
        # heavy atom_set heavy from propane_mol: 0,1,2 + all Hs → old_to_new[0]=0, [1]=1, [2]=2 (Hs start at 3)
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(propane_mol, (0, 1, 2))
        local = tuple(old_to_new[i] for i in (0, 1, 2))
        structure = _make_structure_with_angle(local[0], local[1], local[2], force_k=50.0, theteq=109.5)

        strategy = MinimalMoleculeStrategy()
        context = _minimal_context(propane_mol)

        with patch.object(strategy, "_parameterise", return_value=structure):
            result = strategy.resolve((0, 1, 2), AngleParameter.FORCE_CONSTANT, context)

        assert pytest.approx(result, abs=1e-6) == 50.0

    def test_resolve_raises_for_atom_parameter(self, propane_mol):
        strategy = MinimalMoleculeStrategy()
        context = _minimal_context(propane_mol)
        with pytest.raises(MissingParameterError):
            strategy.resolve((0,), AtomParameter.MASS, context)


class TestReadParameterEdgeCases:
    def test_atom_parameter_raises_missing_error(self):
        # _read_parameter is static; calling it directly with AtomParameter triggers
        # the final fallthrough guard (unreachable from resolve but testable directly)
        structure = _make_structure_with_bond(0, 1, k=1.0, req=1.0)
        with pytest.raises(MissingParameterError, match="Unsupported"):
            MinimalMoleculeStrategy._read_parameter(structure, (0,), AtomParameter.CHARGE)


class TestParameteriseMethod:
    def test_parameterise_calls_conformer_obabel_acpype(self, propane_mol):
        from unittest.mock import MagicMock, patch

        mock_structure = pmd.Structure()
        mock_structure.add_atom(pmd.Atom(), "MOL", 1)

        mol, _ = MinimalMoleculeStrategy._extract_minimal_mol(propane_mol, (0, 1))

        mock_conformer = MagicMock()
        mock_conformer.embed.return_value = mol
        strategy = MinimalMoleculeStrategy(conformer_generator=mock_conformer)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("polymer_md.parameterisation.strategies.minimal_molecule.RDKitHelper") as mock_rdkit,
                patch("polymer_md.parameterisation.strategies.minimal_molecule.OBabelConverter") as MockOBabel,
                patch("polymer_md.parameterisation.strategies.minimal_molecule.AcpypeConverter") as MockAcpype,
                patch("polymer_md.parameterisation.strategies.minimal_molecule.pmd") as mock_pmd,
            ):
                mock_mol2 = tmp_path / "minimal.mol2"
                mock_mol2.touch()
                MockOBabel.return_value.convert.return_value = mock_mol2

                from polymer_md.conversion.gromacs_files import GromacsFiles
                gro = tmp_path / "minimal.gro"
                top = tmp_path / "minimal.top"
                itp = tmp_path / "minimal.itp"
                for f in [gro, top, itp]:
                    f.write_text("")
                mock_gromacs = GromacsFiles(itp=itp, gro=gro, top=top)
                MockAcpype.return_value.convert.return_value = mock_gromacs
                mock_pmd.load_file.return_value = mock_structure

                result = strategy._parameterise(mol, tmp_path)

        mock_conformer.embed.assert_called_once_with(mol)
        mock_rdkit.write_sdf.assert_called_once()
        MockOBabel.return_value.convert.assert_called_once()
        MockAcpype.return_value.convert.assert_called_once()
        mock_pmd.load_file.assert_called_once()
        assert result is mock_structure
