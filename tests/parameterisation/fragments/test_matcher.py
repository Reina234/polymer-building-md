from __future__ import annotations

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
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    DihedralTerm,
)
from polymer_md.parameterisation.fragments.matching.matcher import FragmentMatcher
from polymer_md.utils.topology_builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Fixtures: parameterised propane structure
# ---------------------------------------------------------------------------

@pytest.fixture
def propane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def propane_structure(propane_mol) -> pmd.Structure:
    structure = TopologyBuilder.build(propane_mol)
    for atom in structure.atoms:
        atom.charge = -0.10
        atom.epsilon = 0.05
        atom.sigma = 0.35
        atom.mass = 12.0
    for bond in structure.bonds:
        bond.type = pmd.BondType(k=300.0, req=1.54)
    for angle in structure.angles:
        angle.type = pmd.AngleType(k=50.0, theteq=109.5)
    for dihedral in structure.dihedrals:
        dihedral.type = pmd.DihedralType(phi_k=0.5, phase=0.0, per=3)
    return structure


@pytest.fixture
def butane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def butane_structure(butane_mol) -> pmd.Structure:
    structure = TopologyBuilder.build(butane_mol)
    for atom in structure.atoms:
        atom.charge = -0.05
        atom.epsilon = 0.1
        atom.sigma = 0.3
        atom.mass = 12.0
    for bond in structure.bonds:
        bond.type = pmd.BondType(k=250.0, req=1.52)
    for angle in structure.angles:
        angle.type = pmd.AngleType(k=45.0, theteq=111.0)
    for dihedral in structure.dihedrals:
        dihedral.type = pmd.DihedralType(phi_k=0.8, phase=0.0, per=2)
    return structure


# ---------------------------------------------------------------------------
# Tests: match_all
# ---------------------------------------------------------------------------

class TestMatchAll:
    def test_returns_matches_for_valid_fragment(self, propane_mol):
        fragment = Fragment(
            pattern="[#6;A]-[#6;A]",
            annotated_bonds=(AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        assert len(matches) > 0

    def test_returns_empty_for_no_match(self, propane_mol):
        fragment = Fragment(
            pattern="[Si]",
            annotated_atoms=(AnnotatedAtom(local_index=0, parameter=AtomParameter.CHARGE),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        assert matches == []

    def test_multiple_fragments_combined(self, propane_mol):
        f1 = Fragment(
            pattern="[#6;A]-[#6;A]",
            annotated_bonds=(AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT),),
        )
        f2 = Fragment(
            pattern="[#6;A]",
            annotated_atoms=(AnnotatedAtom(local_index=0, parameter=AtomParameter.CHARGE),),
        )
        matcher = FragmentMatcher(fragments=[f1, f2])
        matches = matcher.match_all(propane_mol)
        fragments_seen = {m.fragment for m in matches}
        assert f1 in fragments_seen
        assert f2 in fragments_seen


# ---------------------------------------------------------------------------
# Tests: build_records
# ---------------------------------------------------------------------------

class TestBuildRecords:
    def test_produces_record_for_each_unique_key(self, propane_mol, propane_structure):
        fragment = Fragment(
            pattern="[#6;A]-[#6;A]",
            annotated_bonds=(AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        records = matcher.build_records(matches, propane_structure)
        assert len(records) > 0

    def test_record_parameter_matches_member(self, propane_mol, propane_structure):
        fragment = Fragment(
            pattern="[#6;A]-[#6;A]",
            annotated_bonds=(AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        records = matcher.build_records(matches, propane_structure)
        for record in records:
            assert record.parameter == BondParameter.FORCE_CONSTANT

    def test_hit_values_match_structure(self, propane_mol, propane_structure):
        fragment = Fragment(
            pattern="[#6;A]",
            annotated_atoms=(AnnotatedAtom(local_index=0, parameter=AtomParameter.CHARGE),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        records = matcher.build_records(matches, propane_structure)
        for record in records:
            if record.parameter == AtomParameter.CHARGE:
                for hit in record.hits:
                    assert pytest.approx(hit.value, abs=1e-6) == -0.10

    def test_empty_matches_give_empty_records(self):
        matcher = FragmentMatcher(fragments=[])
        records = matcher.build_records([], pmd.Structure())
        assert records == []


# ---------------------------------------------------------------------------
# Tests: _local_indices_of
# ---------------------------------------------------------------------------

class TestLocalIndicesOf:
    def test_atom_returns_singleton_tuple(self):
        member = AnnotatedAtom(local_index=5, parameter=AtomParameter.CHARGE)
        assert FragmentMatcher._local_indices_of(member) == (5,)

    def test_bond_returns_pair(self):
        member = AnnotatedBond(local_indices=(2, 3), parameter=BondParameter.FORCE_CONSTANT)
        assert FragmentMatcher._local_indices_of(member) == (2, 3)

    def test_angle_returns_triple(self):
        member = AnnotatedAngle(local_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT)
        assert FragmentMatcher._local_indices_of(member) == (0, 1, 2)

    def test_dihedral_returns_quad(self):
        member = AnnotatedDihedral(local_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT)
        assert FragmentMatcher._local_indices_of(member) == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Tests: _extract_value for each member type
# ---------------------------------------------------------------------------

class TestExtractValue:
    def test_bond_force_constant(self, propane_mol, propane_structure):
        # Find an actual bond in propane
        bond = propane_structure.bonds[0]
        i, j = bond.atom1.idx, bond.atom2.idx
        member = AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT)
        result = FragmentMatcher._extract_value(propane_structure, (i, j), member)
        assert pytest.approx(result, abs=1e-6) == 300.0

    def test_bond_equilibrium_length(self, propane_mol, propane_structure):
        bond = propane_structure.bonds[0]
        i, j = bond.atom1.idx, bond.atom2.idx
        member = AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.EQUILIBRIUM_LENGTH)
        result = FragmentMatcher._extract_value(propane_structure, (i, j), member)
        assert pytest.approx(result, abs=1e-6) == 1.54

    def test_angle_force_constant(self, propane_mol, propane_structure):
        angle = propane_structure.angles[0]
        i, j, k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
        member = AnnotatedAngle(local_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT)
        result = FragmentMatcher._extract_value(propane_structure, (i, j, k), member)
        assert pytest.approx(result, abs=1e-6) == 50.0

    def test_angle_equilibrium_angle(self, propane_mol, propane_structure):
        angle = propane_structure.angles[0]
        i, j, k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
        member = AnnotatedAngle(local_indices=(0, 1, 2), parameter=AngleParameter.EQUILIBRIUM_ANGLE)
        result = FragmentMatcher._extract_value(propane_structure, (i, j, k), member)
        assert pytest.approx(result, abs=1e-6) == 109.5

    def test_dihedral_returns_all_terms(self, butane_mol, butane_structure):
        dihedral = butane_structure.dihedrals[0]
        i, j, k, l = (
            dihedral.atom1.idx, dihedral.atom2.idx,
            dihedral.atom3.idx, dihedral.atom4.idx,
        )
        member = AnnotatedDihedral(local_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT)
        result = FragmentMatcher._extract_value(butane_structure, (i, j, k, l), member)
        assert isinstance(result, tuple)
        assert len(result) >= 1
        assert isinstance(result[0], DihedralTerm)
        assert pytest.approx(result[0].force_constant, abs=1e-6) == 0.8
        assert pytest.approx(result[0].phase, abs=1e-6) == 0.0
        assert pytest.approx(result[0].periodicity, abs=1e-6) == 2.0

    def test_atom_charge(self, propane_mol, propane_structure):
        member = AnnotatedAtom(local_index=0, parameter=AtomParameter.CHARGE)
        result = FragmentMatcher._extract_value(propane_structure, (0,), member)
        assert pytest.approx(result, abs=1e-6) == -0.10

    def test_atom_epsilon(self, propane_mol, propane_structure):
        member = AnnotatedAtom(local_index=0, parameter=AtomParameter.EPSILON)
        result = FragmentMatcher._extract_value(propane_structure, (0,), member)
        assert pytest.approx(result, abs=1e-6) == 0.05

    def test_atom_sigma(self, propane_mol, propane_structure):
        member = AnnotatedAtom(local_index=0, parameter=AtomParameter.SIGMA)
        result = FragmentMatcher._extract_value(propane_structure, (0,), member)
        assert pytest.approx(result, abs=1e-6) == 0.35

    def test_atom_mass(self, propane_mol, propane_structure):
        member = AnnotatedAtom(local_index=0, parameter=AtomParameter.MASS)
        result = FragmentMatcher._extract_value(propane_structure, (0,), member)
        assert pytest.approx(result, abs=1e-6) == 12.0


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------

class TestExtractValueErrors:
    def test_bond_not_found_raises(self, propane_structure):
        member = AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT)
        with pytest.raises(ValueError, match="Bond not found"):
            FragmentMatcher._extract_value(propane_structure, (99, 100), member)

    def test_angle_not_found_raises(self, propane_structure):
        member = AnnotatedAngle(local_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT)
        with pytest.raises(ValueError, match="Angle not found"):
            FragmentMatcher._extract_value(propane_structure, (99, 100, 101), member)

    def test_dihedral_not_found_raises(self, butane_structure):
        member = AnnotatedDihedral(local_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT)
        with pytest.raises(ValueError, match="dihedral not found"):
            FragmentMatcher._extract_value(butane_structure, (99, 100, 101, 102), member)


# ---------------------------------------------------------------------------
# Tests: end-to-end build_records produces correct values
# ---------------------------------------------------------------------------

class TestBuildRecordsEndToEnd:
    def test_bond_record_values_are_correct(self, propane_mol, propane_structure):
        fragment = Fragment(
            pattern="[#6;A]-[#6;A]",
            annotated_bonds=(AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        records = matcher.build_records(matches, propane_structure)

        all_hit_values = [hit.value for r in records for hit in r.hits]
        assert all(pytest.approx(v, abs=1e-6) == 300.0 for v in all_hit_values)
        assert len(all_hit_values) > 0

    def test_atom_record_global_indices_match_structure(self, propane_mol, propane_structure):
        # Atom charge record: global index should be within range of structure atoms
        fragment = Fragment(
            pattern="[#6;A]",
            annotated_atoms=(AnnotatedAtom(local_index=0, parameter=AtomParameter.CHARGE),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        records = matcher.build_records(matches, propane_structure)

        n_atoms = len(propane_structure.atoms)
        for record in records:
            for idx in record.global_indices:
                assert 0 <= idx < n_atoms

    def test_angle_records_cover_all_angles(self, propane_mol, propane_structure):
        fragment = Fragment(
            pattern="[#6;A]-[#6;A]-[#6;A]",
            annotated_angles=(AnnotatedAngle(local_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT),),
        )
        matcher = FragmentMatcher(fragments=[fragment])
        matches = matcher.match_all(propane_mol)
        records = matcher.build_records(matches, propane_structure)

        # Check that values are correct
        for record in records:
            if record.parameter == AngleParameter.FORCE_CONSTANT:
                for hit in record.hits:
                    assert pytest.approx(hit.value, abs=1e-6) == 50.0


# ---------------------------------------------------------------------------
# Tests: _dihedral_terms — multi-periodicity extraction
# ---------------------------------------------------------------------------

class TestDihedralTerms:
    def _make_structure_with_dihedral(
        self, phi_k: float, phase: float, per: float
    ) -> pmd.Structure:
        structure = pmd.Structure()
        atoms = [pmd.Atom() for _ in range(4)]
        for atom in atoms:
            structure.add_atom(atom, "MOL", 1)
        dihedral = pmd.Dihedral(atoms[0], atoms[1], atoms[2], atoms[3])
        dihedral.type = pmd.DihedralType(phi_k=phi_k, phase=phase, per=per)
        structure.dihedrals.append(dihedral)
        return structure

    def test_single_term_returns_one_element_tuple(self):
        structure = self._make_structure_with_dihedral(phi_k=1.5, phase=0.0, per=2.0)
        result = FragmentMatcher._dihedral_terms(structure, (0, 1, 2, 3))
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], DihedralTerm)
        assert pytest.approx(result[0].force_constant, abs=1e-6) == 1.5
        assert pytest.approx(result[0].phase, abs=1e-6) == 0.0
        assert pytest.approx(result[0].periodicity, abs=1e-6) == 2.0

    def test_type_list_returns_all_terms(self):
        from parmed.topologyobjects import DihedralType, DihedralTypeList
        structure = pmd.Structure()
        atoms = [pmd.Atom() for _ in range(4)]
        for atom in atoms:
            structure.add_atom(atom, "MOL", 1)
        dtype_list = DihedralTypeList()
        dtype_list.append(DihedralType(phi_k=1.0, phase=0.0, per=1))
        dtype_list.append(DihedralType(phi_k=0.5, phase=3.14, per=3))
        dihedral = pmd.Dihedral(atoms[0], atoms[1], atoms[2], atoms[3])
        dihedral.type = dtype_list
        structure.dihedrals.append(dihedral)
        result = FragmentMatcher._dihedral_terms(structure, (0, 1, 2, 3))
        assert len(result) == 2
        assert pytest.approx(result[0].force_constant, abs=1e-6) == 1.0
        assert pytest.approx(result[1].force_constant, abs=1e-6) == 0.5
        assert pytest.approx(result[1].phase, abs=1e-3) == 3.14

    def test_dihedral_not_found_raises(self):
        structure = self._make_structure_with_dihedral(phi_k=1.0, phase=0.0, per=2.0)
        with pytest.raises(ValueError, match="dihedral not found"):
            FragmentMatcher._dihedral_terms(structure, (99, 100, 101, 102))
