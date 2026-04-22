from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterHit, ParameterRecord
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.fragments.library import FragmentLibrary


_PATTERN = "[#6;A]-[#6;A]"


class TestFragmentLibraryAtomMetadata:
    def test_atom_metadata_stored_on_construction(self, library_with_metadata):
        # arrange / act done by fixture

        # assert
        assert _PATTERN in library_with_metadata.atom_metadata
        assert 0 in library_with_metadata.atom_metadata[_PATTERN]
        assert library_with_metadata.atom_metadata[_PATTERN][0].gaff2_type == "c3"

    def test_metadata_for_helper(self, library_with_metadata):
        # arrange
        pattern = _PATTERN

        # act
        meta = library_with_metadata.metadata_for(pattern, 0)

        # assert
        assert meta is not None
        assert meta.residue_id == "S"
        assert meta.within_residue_position == 0

    def test_metadata_for_returns_none_for_missing_pattern(self, library_with_metadata):
        # arrange / act
        meta = library_with_metadata.metadata_for("nonexistent_pattern", 0)

        # assert
        assert meta is None

    def test_metadata_for_returns_none_for_missing_local_idx(self, library_with_metadata):
        # arrange / act
        meta = library_with_metadata.metadata_for(_PATTERN, 99)

        # assert
        assert meta is None


class TestFragmentLibrarySerialisation:
    def test_round_trip_preserves_records(self, library_with_metadata):
        # arrange
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        # act
        library_with_metadata.save(path)
        loaded = FragmentLibrary.load(path)

        # assert
        assert len(loaded.records) == len(library_with_metadata.records)

    def test_round_trip_preserves_atom_metadata(self, library_with_metadata):
        # arrange
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        # act
        library_with_metadata.save(path)
        loaded = FragmentLibrary.load(path)

        # assert
        assert _PATTERN in loaded.atom_metadata
        assert loaded.atom_metadata[_PATTERN][0].gaff2_type == "c3"
        assert loaded.atom_metadata[_PATTERN][0].residue_id == "S"
        assert loaded.atom_metadata[_PATTERN][0].within_residue_position == 0
        assert loaded.atom_metadata[_PATTERN][1].residue_id == "S"
        assert loaded.atom_metadata[_PATTERN][1].within_residue_position == 1

    def test_round_trip_local_idx_keys_are_ints(self, library_with_metadata):
        # JSON stores dict keys as strings; loading must convert back to int

        # arrange
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        library_with_metadata.save(path)
        loaded = FragmentLibrary.load(path)

        # act
        keys = list(loaded.atom_metadata[_PATTERN].keys())

        # assert – keys must be ints, not strings
        assert all(isinstance(k, int) for k in keys)

    def test_round_trip_empty_metadata(self):
        # arrange
        library = FragmentLibrary(records=(), atom_metadata={})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        # act
        library.save(path)
        loaded = FragmentLibrary.load(path)

        # assert
        assert loaded.atom_metadata == {}

    def test_json_structure_contains_both_sections(self, library_with_metadata):
        # arrange
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        library_with_metadata.save(path)

        # act
        raw = json.loads(path.read_text())

        # assert
        assert "records" in raw
        assert "atom_metadata" in raw

    def test_round_trip_preserves_parameter_values(self, library_with_metadata):
        # arrange
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        library_with_metadata.save(path)

        # act
        loaded = FragmentLibrary.load(path)
        charge_records = [r for r in loaded.records if r.parameter == AtomParameter.CHARGE]

        # assert
        assert len(charge_records) > 0
        assert all(r.hits for r in charge_records)


class TestFragmentLibraryHitValuesQuery:
    def test_returns_values_for_matching_residue_position(self, library_with_metadata):
        # arrange / act
        values = library_with_metadata.hit_values_for_residue_position(
            residue_id="S",
            within_residue_position=0,
            parameter=AtomParameter.CHARGE,
        )

        # assert – should find both _PATTERN_CC (pos 0 = -0.10) and _PATTERN_CCC (pos 0 = -0.11)
        assert len(values) == 2
        assert pytest.approx(-0.10, abs=1e-6) in values
        assert pytest.approx(-0.11, abs=1e-6) in values

    def test_returns_empty_for_unknown_residue(self, library_with_metadata):
        # arrange / act
        values = library_with_metadata.hit_values_for_residue_position(
            residue_id="UNKNOWN",
            within_residue_position=0,
            parameter=AtomParameter.CHARGE,
        )

        # assert
        assert values == []

    def test_returns_empty_for_wrong_parameter_type(self, library_with_metadata):
        # arrange / act
        values = library_with_metadata.hit_values_for_residue_position(
            residue_id="S",
            within_residue_position=0,
            parameter=AtomParameter.EPSILON,
        )

        # assert – no EPSILON records in fixture
        assert values == []

    def test_does_not_return_bond_records(self, library_with_metadata):
        # The fixture has a BondParameter record; the atom query must not return it

        # arrange / act
        values = library_with_metadata.hit_values_for_residue_position(
            residue_id="S",
            within_residue_position=0,
            parameter=BondParameter.FORCE_CONSTANT,
        )

        # assert – BondParameter record has multi-index hits, should be excluded
        assert values == []
