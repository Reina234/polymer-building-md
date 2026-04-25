from __future__ import annotations

import pytest

from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterHit, ParameterRecord
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.fragments.library import FragmentLibrary

_PATTERN = "[#6;A]-[#6;A]"


def _make_library() -> FragmentLibrary:
    fragment = Fragment(pattern=_PATTERN)
    records = (
        ParameterRecord(
            global_indices=(0,),
            parameter=AtomParameter.CHARGE,
            hits=(
                ParameterHit(value=-0.10, fragment=fragment, match_instance=0, member_local_indices=(0,)),
                ParameterHit(value=-0.12, fragment=fragment, match_instance=1, member_local_indices=(0,)),
            ),
        ),
        ParameterRecord(
            global_indices=(0, 1),
            parameter=BondParameter.FORCE_CONSTANT,
            hits=(
                ParameterHit(value=300.0, fragment=fragment, match_instance=0, member_local_indices=(0, 1)),
            ),
        ),
    )
    return FragmentLibrary(records=records, atom_metadata={})


class TestValuesForMember:
    def test_returns_all_matching_hit_values(self):
        library = _make_library()
        values = library.values_for_member(_PATTERN, (0,), AtomParameter.CHARGE)
        assert sorted(values) == pytest.approx(sorted([-0.10, -0.12]))

    def test_returns_empty_for_wrong_parameter(self):
        library = _make_library()
        values = library.values_for_member(_PATTERN, (0,), AtomParameter.EPSILON)
        assert values == []

    def test_returns_empty_for_wrong_pattern(self):
        library = _make_library()
        values = library.values_for_member("[#6;A]", (0,), AtomParameter.CHARGE)
        assert values == []

    def test_returns_empty_for_wrong_local_indices(self):
        library = _make_library()
        values = library.values_for_member(_PATTERN, (1,), AtomParameter.CHARGE)
        assert values == []

    def test_returns_bond_values(self):
        library = _make_library()
        values = library.values_for_member(_PATTERN, (0, 1), BondParameter.FORCE_CONSTANT)
        assert values == pytest.approx([300.0])

    def test_empty_library_returns_empty(self):
        library = FragmentLibrary(records=(), atom_metadata={})
        values = library.values_for_member(_PATTERN, (0,), AtomParameter.CHARGE)
        assert values == []
