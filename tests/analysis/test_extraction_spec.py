from __future__ import annotations

import pytest

from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle, AnnotatedAtom, AnnotatedBond, AnnotatedDihedral,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter, AtomParameter, BondParameter, DihedralParameter,
)


class TestExtractionSpecToFragment:
    def test_invalid_pattern_raises(self):
        spec = ExtractionSpec(pattern="not_valid_smarts!!!", parameters=())
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            spec.to_fragment()

    def test_pattern_preserved_in_fragment(self):
        spec = ExtractionSpec(pattern="[#6;A]-[#6;A]", parameters=(BondParameter.FORCE_CONSTANT,))
        fragment = spec.to_fragment()
        assert fragment.pattern == "[#6;A]-[#6;A]"

    def test_bond_parameters_enumerated(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]-[#6;A]",
            parameters=(BondParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_bonds) == 2
        assert all(isinstance(b, AnnotatedBond) for b in fragment.annotated_bonds)
        assert all(b.parameter == BondParameter.FORCE_CONSTANT for b in fragment.annotated_bonds)

    def test_both_bond_parameters_enumerated(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]",
            parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
        )
        fragment = spec.to_fragment()
        params = {b.parameter for b in fragment.annotated_bonds}
        assert BondParameter.FORCE_CONSTANT in params
        assert BondParameter.EQUILIBRIUM_LENGTH in params

    def test_angle_parameters_enumerated(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]-[#6;A]",
            parameters=(AngleParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_angles) == 1
        assert fragment.annotated_angles[0].parameter == AngleParameter.FORCE_CONSTANT
        assert fragment.annotated_angles[0].local_indices == (0, 1, 2)

    def test_dihedral_parameters_enumerated(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]-[#6;A]-[#6;A]",
            parameters=(DihedralParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_dihedrals) >= 1
        assert all(isinstance(d, AnnotatedDihedral) for d in fragment.annotated_dihedrals)

    def test_atom_parameters_enumerated(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]",
            parameters=(AtomParameter.CHARGE,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_atoms) == 2
        assert all(isinstance(a, AnnotatedAtom) for a in fragment.annotated_atoms)
        assert all(a.parameter == AtomParameter.CHARGE for a in fragment.annotated_atoms)

    def test_no_bonds_when_no_bond_parameters(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]",
            parameters=(AtomParameter.CHARGE,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_bonds) == 0

    def test_no_angles_when_only_two_atoms(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]",
            parameters=(AngleParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_angles) == 0

    def test_no_dihedrals_when_three_atoms(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]-[#6;A]",
            parameters=(DihedralParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_dihedrals) == 0

    def test_bond_local_indices_are_sorted(self):
        spec = ExtractionSpec(
            pattern="[#6;A]-[#6;A]",
            parameters=(BondParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        for bond in fragment.annotated_bonds:
            i, j = bond.local_indices
            assert i <= j

    def test_no_duplicate_angles(self):
        spec = ExtractionSpec(
            pattern="[#6;A](-[#6;A])-[#6;A]",
            parameters=(AngleParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        keys = [(a.local_indices, a.parameter) for a in fragment.annotated_angles]
        assert len(keys) == len(set(keys))

    def test_empty_parameters_gives_empty_fragment(self):
        spec = ExtractionSpec(pattern="[#6;A]-[#6;A]", parameters=())
        fragment = spec.to_fragment()
        assert len(fragment.all_members) == 0

    def test_cyclopropane_has_no_dihedrals_due_to_shared_neighbors(self):
        spec = ExtractionSpec(
            pattern="[#6]1[#6][#6]1",
            parameters=(DihedralParameter.FORCE_CONSTANT,),
        )
        fragment = spec.to_fragment()
        assert len(fragment.annotated_dihedrals) == 0
