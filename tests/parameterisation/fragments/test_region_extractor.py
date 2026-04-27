from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
)
from polymer_md.parameterisation.fragments.extraction.region_extractor import RegionFragmentExtractor
from polymer_md.utils.topology_builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def propane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def butane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def propane_structure(propane_mol) -> pmd.Structure:
    return TopologyBuilder.build(propane_mol)


@pytest.fixture
def butane_structure(butane_mol) -> pmd.Structure:
    return TopologyBuilder.build(butane_mol)


def _heavy_indices(mol: Chem.Mol) -> frozenset[int]:
    return frozenset(
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1
    )


# ---------------------------------------------------------------------------
# Tests: resolve_parmed_indices
# ---------------------------------------------------------------------------

class TestResolveParmedIndices:
    def test_heavy_indices_only_maps_to_parmed(self, propane_mol, propane_structure):
        heavy = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        # identity crosswalk: molecule built directly from same mol
        mol3d_to_parmed = {i: i for i in range(propane_mol.GetNumAtoms())}
        result = RegionFragmentExtractor.resolve_parmed_indices(
            heavy, propane_mol, mol3d_to_parmed
        )
        # Heavy atoms must be in result
        for idx in heavy:
            assert idx in result

    def test_attached_hydrogens_are_included(self, propane_mol):
        heavy = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        mol3d_to_parmed = {i: i for i in range(propane_mol.GetNumAtoms())}
        result = RegionFragmentExtractor.resolve_parmed_indices(
            heavy, propane_mol, mol3d_to_parmed
        )
        h_indices = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        )
        assert len(result & h_indices) > 0

    def test_unmapped_indices_excluded(self, propane_mol):
        heavy = frozenset([0, 1])
        # mol3d_to_parmed only covers atom 0 — atom 1 not mapped
        mol3d_to_parmed = {0: 0}
        result = RegionFragmentExtractor.resolve_parmed_indices(
            heavy, propane_mol, mol3d_to_parmed
        )
        # atom 1 was not mapped, so 1 should not appear in result
        assert 1 not in result


# ---------------------------------------------------------------------------
# Tests: _find_attached_hydrogen_indices
# ---------------------------------------------------------------------------

class TestFindAttachedHydrogenIndices:
    def test_returns_hydrogens_bonded_to_heavy_atoms(self, propane_mol):
        heavy = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        hydrogens = RegionFragmentExtractor._find_attached_hydrogen_indices(propane_mol, heavy)
        for h_idx in hydrogens:
            atom = propane_mol.GetAtomWithIdx(h_idx)
            assert atom.GetAtomicNum() == 1

    def test_all_h_atoms_attached_to_region_are_found(self, propane_mol):
        heavy = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        hydrogens = RegionFragmentExtractor._find_attached_hydrogen_indices(propane_mol, heavy)
        all_h = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        )
        assert hydrogens == all_h  # all H atoms are attached to some heavy atom in propane

    def test_empty_when_no_heavy_neighbours(self, propane_mol):
        result = RegionFragmentExtractor._find_attached_hydrogen_indices(propane_mol, frozenset())
        assert result == frozenset()


# ---------------------------------------------------------------------------
# Tests: extract — atoms
# ---------------------------------------------------------------------------

class TestExtractAtomMembers:
    def test_region_atoms_annotated_for_all_atom_parameters(self, propane_mol, propane_structure):
        heavy = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        all_atoms = frozenset(range(propane_mol.GetNumAtoms()))

        extractor = RegionFragmentExtractor()
        fragment, global_to_local = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=all_atoms,
            region_parmed_indices=heavy,
        )
        atom_params_found = {m.parameter for m in fragment.annotated_atoms}
        for param in AtomParameter:
            assert param in atom_params_found

    def test_non_region_atoms_not_annotated(self, propane_mol, propane_structure):
        # Use only atom 0 as the region; atoms 1 and 2 should not appear
        region = frozenset([0])
        context = frozenset(range(propane_mol.GetNumAtoms()))
        extractor = RegionFragmentExtractor()
        fragment, global_to_local = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        region_locals = {global_to_local[i] for i in region if i in global_to_local}
        for member in fragment.annotated_atoms:
            assert member.local_index in region_locals


# ---------------------------------------------------------------------------
# Tests: extract — bonds
# ---------------------------------------------------------------------------

class TestExtractBondMembers:
    def test_bonds_touching_region_are_included(self, propane_mol, propane_structure):
        # Region = middle carbon (index 1); bonds touching it include C0-C1 and C1-C2
        middle_c = 1
        region = frozenset([middle_c])
        context = frozenset(range(propane_mol.GetNumAtoms()))
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        assert len(fragment.annotated_bonds) > 0

    def test_bond_parameters_cover_all_bond_params(self, propane_mol, propane_structure):
        context = frozenset(range(propane_mol.GetNumAtoms()))
        region = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        bond_params = {m.parameter for m in fragment.annotated_bonds}
        for param in BondParameter:
            assert param in bond_params

    def test_bonds_not_in_context_excluded(self, propane_mol, propane_structure):
        # Context = only atoms 0 and 1; bond (1,2) should not appear since atom 2 is out of context
        context = frozenset([0, 1])
        region = frozenset([0, 1])
        extractor = RegionFragmentExtractor()
        fragment, global_to_local = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        # All bond local indices must be within global_to_local keys
        for member in fragment.annotated_bonds:
            for li in member.local_indices:
                assert li in global_to_local.values()


# ---------------------------------------------------------------------------
# Tests: extract — angles
# ---------------------------------------------------------------------------

class TestExtractAngleMembers:
    def test_angles_with_middle_atom_in_region(self, propane_mol, propane_structure):
        middle_c = 1
        region = frozenset([middle_c])
        context = frozenset(range(propane_mol.GetNumAtoms()))
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        # Middle carbon is the centre of all H-C-C and H-C-H angles
        assert len(fragment.annotated_angles) > 0

    def test_angles_with_no_atom_in_region_excluded(self, propane_mol, propane_structure):
        # Only terminal C (index 0) in region; angles where no atom touches region excluded.
        # Angles spanning into region (atom 0 as endpoint) are now included, consistent
        # with the "touches region" rule used for bonds.
        region = frozenset([0])
        context = frozenset(range(propane_mol.GetNumAtoms()))
        extractor = RegionFragmentExtractor()
        fragment, global_to_local = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        local_to_global = {v: k for k, v in global_to_local.items()}
        for member in fragment.annotated_angles:
            global_indices = tuple(local_to_global[li] for li in member.local_indices)
            assert any(g in region for g in global_indices)


# ---------------------------------------------------------------------------
# Tests: extract — dihedrals
# ---------------------------------------------------------------------------

class TestExtractDihedralMembers:
    def test_dihedrals_present_for_butane(self, butane_mol, butane_structure):
        region = frozenset(
            i for i in range(butane_mol.GetNumAtoms())
            if butane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        context = frozenset(range(butane_mol.GetNumAtoms()))
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            butane_mol, butane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        assert len(fragment.annotated_dihedrals) > 0

    def test_one_annotated_dihedral_per_quadruple(self, butane_mol, butane_structure):
        region = frozenset(
            i for i in range(butane_mol.GetNumAtoms())
            if butane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        context = frozenset(range(butane_mol.GetNumAtoms()))
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            butane_mol, butane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        local_index_sets = {m.local_indices for m in fragment.annotated_dihedrals}
        assert len(fragment.annotated_dihedrals) == len(local_index_sets)
        assert all(m.parameter == DihedralParameter.FORCE_CONSTANT for m in fragment.annotated_dihedrals)

    def test_no_duplicate_dihedrals(self, butane_mol, butane_structure):
        context = frozenset(range(butane_mol.GetNumAtoms()))
        region = frozenset(
            i for i in range(butane_mol.GetNumAtoms())
            if butane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            butane_mol, butane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        seen = set()
        for m in fragment.annotated_dihedrals:
            key = (m.local_indices, m.parameter)
            assert key not in seen, f"Duplicate dihedral: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# Tests: extract — fragment pattern is valid SMARTS
# ---------------------------------------------------------------------------

class TestExtractFragmentPattern:
    def test_fragment_pattern_is_valid_smarts(self, propane_mol, propane_structure):
        context = frozenset(range(propane_mol.GetNumAtoms()))
        region = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        query = Chem.MolFromSmarts(fragment.pattern)
        assert query is not None

    def test_fragment_pattern_matches_mol(self, propane_mol, propane_structure):
        context = frozenset(range(propane_mol.GetNumAtoms()))
        region = frozenset(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        extractor = RegionFragmentExtractor()
        fragment, _ = extractor.extract(
            propane_mol, propane_structure,
            context_parmed_indices=context,
            region_parmed_indices=region,
        )
        query = Chem.MolFromSmarts(fragment.pattern)
        matches = propane_mol.GetSubstructMatches(query)
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# Tests: defensive deduplication and filtering branches
# ---------------------------------------------------------------------------

def _mock_bond(i: int, j: int) -> MagicMock:
    b = MagicMock()
    b.atom1.idx = i
    b.atom2.idx = j
    return b


def _mock_angle(i: int, j: int, k: int) -> MagicMock:
    a = MagicMock()
    a.atom1.idx = i
    a.atom2.idx = j
    a.atom3.idx = k
    return a


def _mock_dihedral(i: int, j: int, k: int, l: int, improper: bool = False) -> MagicMock:
    d = MagicMock()
    d.improper = improper
    d.atom1.idx = i
    d.atom2.idx = j
    d.atom3.idx = k
    d.atom4.idx = l
    return d


class TestExtractAtomMembersDirectly:
    def test_atom_not_in_context_silently_skipped(self):
        extractor = RegionFragmentExtractor()
        region = frozenset({0, 1, 99})
        global_to_local = {0: 0, 1: 1}

        members = extractor._extract_atom_members(region, global_to_local)

        present_locals = {m.local_index for m in members}
        assert 0 in present_locals
        assert 1 in present_locals
        assert len(members) == 2 * len(list(AtomParameter))

    def test_atom_not_in_context_does_not_raise(self):
        extractor = RegionFragmentExtractor()
        region = frozenset({999})
        global_to_local = {}

        members = extractor._extract_atom_members(region, global_to_local)
        assert members == []


class TestExtractBondMembersDirectly:
    def test_duplicate_bond_deduplicated(self):
        extractor = RegionFragmentExtractor()
        bond = _mock_bond(0, 1)
        mock_structure = MagicMock()
        mock_structure.bonds = [bond, bond]

        region = frozenset({0, 1})
        global_to_local = {0: 0, 1: 1}
        members = extractor._extract_bond_members(mock_structure, region, global_to_local)

        local_keys = {m.local_indices for m in members}
        assert len(local_keys) == 1
        assert len(members) == len(list(BondParameter))

    def test_reversed_bond_treated_as_same_bond(self):
        extractor = RegionFragmentExtractor()
        bond_fwd = _mock_bond(0, 1)
        bond_rev = _mock_bond(1, 0)
        mock_structure = MagicMock()
        mock_structure.bonds = [bond_fwd, bond_rev]

        region = frozenset({0, 1})
        global_to_local = {0: 0, 1: 1}
        members = extractor._extract_bond_members(mock_structure, region, global_to_local)

        assert len(members) == len(list(BondParameter))


class TestExtractAngleMembersDirectly:
    def test_duplicate_angle_deduplicated(self):
        extractor = RegionFragmentExtractor()
        angle = _mock_angle(0, 1, 2)
        mock_structure = MagicMock()
        mock_structure.angles = [angle, angle]

        region = frozenset({1})
        global_to_local = {0: 0, 1: 1, 2: 2}
        members = extractor._extract_angle_members(mock_structure, region, global_to_local)

        assert len(members) == len(list(AngleParameter))

    def test_reversed_angle_treated_as_same_angle(self):
        extractor = RegionFragmentExtractor()
        angle_fwd = _mock_angle(0, 1, 2)
        angle_rev = _mock_angle(2, 1, 0)
        mock_structure = MagicMock()
        mock_structure.angles = [angle_fwd, angle_rev]

        region = frozenset({1})
        global_to_local = {0: 0, 1: 1, 2: 2}
        members = extractor._extract_angle_members(mock_structure, region, global_to_local)

        assert len(members) == len(list(AngleParameter))


class TestExtractDihedralMembersDirectly:
    def test_improper_dihedral_skipped(self):
        extractor = RegionFragmentExtractor()
        proper = _mock_dihedral(0, 1, 2, 3, improper=False)
        improper = _mock_dihedral(0, 1, 2, 3, improper=True)
        mock_structure = MagicMock()
        mock_structure.dihedrals = [improper, proper]

        region = frozenset({1, 2})
        global_to_local = {0: 0, 1: 1, 2: 2, 3: 3}
        members = extractor._extract_dihedral_members(mock_structure, region, global_to_local)

        assert len(members) == 1

    def test_only_improper_dihedrals_gives_no_members(self):
        extractor = RegionFragmentExtractor()
        mock_structure = MagicMock()
        mock_structure.dihedrals = [
            _mock_dihedral(0, 1, 2, 3, improper=True),
            _mock_dihedral(0, 1, 2, 4, improper=True),
        ]
        members = extractor._extract_dihedral_members(
            mock_structure, frozenset({1, 2}), {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        )
        assert members == []

    def test_duplicate_proper_dihedral_deduplicated(self):
        extractor = RegionFragmentExtractor()
        dihedral = _mock_dihedral(0, 1, 2, 3, improper=False)
        mock_structure = MagicMock()
        mock_structure.dihedrals = [dihedral, dihedral]

        region = frozenset({1, 2})
        global_to_local = {0: 0, 1: 1, 2: 2, 3: 3}
        members = extractor._extract_dihedral_members(mock_structure, region, global_to_local)

        assert len(members) == 1

    def test_reversed_dihedral_treated_as_duplicate(self):
        extractor = RegionFragmentExtractor()
        fwd = _mock_dihedral(0, 1, 2, 3, improper=False)
        rev = _mock_dihedral(3, 2, 1, 0, improper=False)
        mock_structure = MagicMock()
        mock_structure.dihedrals = [fwd, rev]

        region = frozenset({1, 2})
        global_to_local = {0: 0, 1: 1, 2: 2, 3: 3}
        members = extractor._extract_dihedral_members(mock_structure, region, global_to_local)

        assert len(members) == 1
