from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
)
from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import (
    NoMatchError,
    ParameterHit,
    ParameterRecord,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    DihedralTerm,
)
from polymer_md.parameterisation.fragments.matching.resolution import (
    FirstStrategy,
    MaxStrategy,
    MeanStrategy,
)
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.fragments.migrate_v1_to_v2 import migrate_v1_to_v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATTERN = "[#6;A]-[#6;A]"
_PATTERN2 = "[#6;A]-[#6;A]-[#6;A]"


def _make_full_library() -> FragmentLibrary:
    """Library with one record of each parameter type for round-trip testing."""
    frag = Fragment(pattern=_PATTERN)
    frag2 = Fragment(pattern=_PATTERN2)

    atom_hit = ParameterHit(value=-0.10, fragment=frag, match_instance=0, member_local_indices=(0,))
    bond_hit = ParameterHit(value=300.0, fragment=frag, match_instance=0, member_local_indices=(0, 1))
    angle_hit = ParameterHit(value=50.0, fragment=frag2, match_instance=0, member_local_indices=(0, 1, 2))
    dihedral_hit = ParameterHit(value=0.5, fragment=frag2, match_instance=0, member_local_indices=(0, 1, 2, 3))

    records = (
        ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(atom_hit,)),
        ParameterRecord(global_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT, hits=(bond_hit,)),
        ParameterRecord(global_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT, hits=(angle_hit,)),
        ParameterRecord(global_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT, hits=(dihedral_hit,)),
    )
    metadata = {
        _PATTERN: {
            0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0),
            1: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=1),
        }
    }
    return FragmentLibrary(records=records, atom_metadata=metadata)


# ---------------------------------------------------------------------------
# Tests: save/load round-trip
# ---------------------------------------------------------------------------

class TestLibraryRoundTrip:
    def test_save_creates_file(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            assert path.exists()

    def test_saved_file_is_valid_json(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            data = json.loads(path.read_text())
            assert "records" in data
            assert "atom_metadata" in data

    def test_round_trip_record_count(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        assert len(loaded.records) == len(library.records)

    def test_round_trip_atom_metadata_preserved(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        assert _PATTERN in loaded.atom_metadata
        meta = loaded.atom_metadata[_PATTERN][0]
        assert meta.gaff2_type == "c3"
        assert meta.residue_id == "S"
        assert meta.within_residue_position == 0

    def test_round_trip_charge_value(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        charge_records = [r for r in loaded.records if r.parameter == AtomParameter.CHARGE]
        assert len(charge_records) == 1
        assert pytest.approx(charge_records[0].hits[0].value, abs=1e-9) == -0.10

    def test_round_trip_bond_value(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        bond_records = [r for r in loaded.records if r.parameter == BondParameter.FORCE_CONSTANT]
        assert len(bond_records) == 1
        assert pytest.approx(bond_records[0].hits[0].value, abs=1e-9) == 300.0

    def test_round_trip_angle_value(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        angle_records = [r for r in loaded.records if r.parameter == AngleParameter.FORCE_CONSTANT]
        assert len(angle_records) == 1
        assert pytest.approx(angle_records[0].hits[0].value, abs=1e-9) == 50.0

    def test_round_trip_dihedral_value(self):
        library = _make_full_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        dihedral_records = [r for r in loaded.records if r.parameter == DihedralParameter.FORCE_CONSTANT]
        assert len(dihedral_records) == 1
        assert pytest.approx(dihedral_records[0].hits[0].value, abs=1e-9) == 0.5

    def test_round_trip_empty_library(self):
        library = FragmentLibrary(records=(), atom_metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        assert len(loaded.records) == 0
        assert loaded.atom_metadata == {}


# ---------------------------------------------------------------------------
# Tests: hit_values_for_residue_position
# ---------------------------------------------------------------------------

class TestHitValuesForResiduePosition:
    def test_returns_values_for_matching_residue_and_position(self):
        frag = Fragment(pattern=_PATTERN)
        hit = ParameterHit(value=-0.10, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        metadata = {
            _PATTERN: {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)}
        }
        library = FragmentLibrary(records=(record,), atom_metadata=metadata)

        values = library.hit_values_for_residue_position("S", 0, AtomParameter.CHARGE)
        assert len(values) == 1
        assert pytest.approx(values[0], abs=1e-9) == -0.10

    def test_returns_empty_for_wrong_residue(self):
        frag = Fragment(pattern=_PATTERN)
        hit = ParameterHit(value=-0.10, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        metadata = {_PATTERN: {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)}}
        library = FragmentLibrary(records=(record,), atom_metadata=metadata)

        values = library.hit_values_for_residue_position("UNKNOWN", 0, AtomParameter.CHARGE)
        assert values == []

    def test_returns_empty_for_wrong_position(self):
        frag = Fragment(pattern=_PATTERN)
        hit = ParameterHit(value=-0.10, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        metadata = {_PATTERN: {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)}}
        library = FragmentLibrary(records=(record,), atom_metadata=metadata)

        values = library.hit_values_for_residue_position("S", 99, AtomParameter.CHARGE)
        assert values == []

    def test_multi_local_index_hits_excluded(self):
        # Multi-index hits (bonds etc.) must not be returned by hit_values_for_residue_position
        frag = Fragment(pattern=_PATTERN)
        bond_hit = ParameterHit(value=300.0, fragment=frag, match_instance=0, member_local_indices=(0, 1))
        record = ParameterRecord(global_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT, hits=(bond_hit,))
        metadata = {
            _PATTERN: {
                0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0),
                1: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=1),
            }
        }
        library = FragmentLibrary(records=(record,), atom_metadata=metadata)

        values = library.hit_values_for_residue_position("S", 0, BondParameter.FORCE_CONSTANT)
        assert values == []

    def test_multiple_contexts_aggregated(self):
        frag1 = Fragment(pattern=_PATTERN)
        frag2 = Fragment(pattern=_PATTERN2)
        hit1 = ParameterHit(value=-0.10, fragment=frag1, match_instance=0, member_local_indices=(0,))
        hit2 = ParameterHit(value=-0.12, fragment=frag2, match_instance=0, member_local_indices=(0,))
        records = (
            ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit1,)),
            ParameterRecord(global_indices=(5,), parameter=AtomParameter.CHARGE, hits=(hit2,)),
        )
        metadata = {
            _PATTERN: {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)},
            _PATTERN2: {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)},
        }
        library = FragmentLibrary(records=records, atom_metadata=metadata)

        values = library.hit_values_for_residue_position("S", 0, AtomParameter.CHARGE)
        assert len(values) == 2
        assert pytest.approx(sorted(values), abs=1e-9) == sorted([-0.10, -0.12])


# ---------------------------------------------------------------------------
# Tests: to_fragments
# ---------------------------------------------------------------------------

class TestToFragments:
    def test_record_with_no_hits_excluded(self):
        record_no_hits = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=())
        library = FragmentLibrary(records=(record_no_hits,), atom_metadata={})
        fragments = library.to_fragments()
        assert fragments == []

    def test_reconstructs_bond_member(self):
        frag = Fragment(pattern=_PATTERN)
        hit = ParameterHit(value=300.0, fragment=frag, match_instance=0, member_local_indices=(0, 1))
        record = ParameterRecord(global_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT, hits=(hit,))
        library = FragmentLibrary(records=(record,), atom_metadata={})
        fragments = library.to_fragments()
        assert len(fragments) == 1
        assert any(isinstance(m, AnnotatedBond) for m in fragments[0].all_members)

    def test_reconstructs_angle_member(self):
        frag = Fragment(pattern=_PATTERN2)
        hit = ParameterHit(value=50.0, fragment=frag, match_instance=0, member_local_indices=(0, 1, 2))
        record = ParameterRecord(global_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT, hits=(hit,))
        library = FragmentLibrary(records=(record,), atom_metadata={})
        fragments = library.to_fragments()
        assert any(isinstance(m, AnnotatedAngle) for m in fragments[0].all_members)

    def test_reconstructs_dihedral_member(self):
        frag = Fragment(pattern=_PATTERN2)
        hit = ParameterHit(value=0.5, fragment=frag, match_instance=0, member_local_indices=(0, 1, 2, 3))
        record = ParameterRecord(global_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT, hits=(hit,))
        library = FragmentLibrary(records=(record,), atom_metadata={})
        fragments = library.to_fragments()
        assert any(isinstance(m, AnnotatedDihedral) for m in fragments[0].all_members)

    def test_deduplicates_same_pattern_member(self):
        frag = Fragment(pattern=_PATTERN)
        hit1 = ParameterHit(value=300.0, fragment=frag, match_instance=0, member_local_indices=(0, 1))
        hit2 = ParameterHit(value=310.0, fragment=frag, match_instance=1, member_local_indices=(0, 1))
        record1 = ParameterRecord(global_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT, hits=(hit1,))
        record2 = ParameterRecord(global_indices=(2, 3), parameter=BondParameter.FORCE_CONSTANT, hits=(hit2,))
        library = FragmentLibrary(records=(record1, record2), atom_metadata={})
        fragments = library.to_fragments()
        assert len(fragments) == 1
        bond_members = [m for m in fragments[0].all_members if isinstance(m, AnnotatedBond)]
        # Same (parameter, local_indices) key → only one member
        assert len(bond_members) == 1


# ---------------------------------------------------------------------------
# Tests: resolution strategies
# ---------------------------------------------------------------------------

class TestLibraryQuery:
    def test_query_returns_matching_record(self):
        library = _make_full_library()
        result = library.query((0,), AtomParameter.CHARGE)
        assert result is not None
        assert result.parameter == AtomParameter.CHARGE
        assert result.global_indices == (0,)

    def test_query_returns_none_when_no_matching_indices(self):
        library = _make_full_library()
        result = library.query((999,), AtomParameter.CHARGE)
        assert result is None

    def test_query_returns_none_for_empty_library(self):
        library = FragmentLibrary(records=(), atom_metadata={})
        result = library.query((0,), AtomParameter.CHARGE)
        assert result is None

    def test_query_matches_on_both_indices_and_parameter(self):
        library = _make_full_library()
        result_charge = library.query((0,), AtomParameter.CHARGE)
        result_bond = library.query((0,), BondParameter.FORCE_CONSTANT)
        assert result_charge is not None
        assert result_bond is None


class TestBuildAnnotatedMembers:
    def test_record_with_hits_produces_member(self):
        frag = Fragment(pattern=_PATTERN)
        hit = ParameterHit(value=-0.1, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        from polymer_md.parameterisation.fragments.library import FragmentLibrary as FL
        members = FL._build_annotated_members([record])
        assert len(members) == 1

    def test_record_with_no_hits_is_skipped(self):
        record_no_hits = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=())
        frag = Fragment(pattern=_PATTERN)
        hit = ParameterHit(value=-0.1, fragment=frag, match_instance=0, member_local_indices=(0,))
        record_with_hit = ParameterRecord(global_indices=(1,), parameter=AtomParameter.CHARGE, hits=(hit,))
        from polymer_md.parameterisation.fragments.library import FragmentLibrary as FL
        members = FL._build_annotated_members([record_no_hits, record_with_hit])
        assert len(members) == 1

    def test_duplicate_parameter_local_indices_deduplicated(self):
        frag = Fragment(pattern=_PATTERN)
        hit1 = ParameterHit(value=1.0, fragment=frag, match_instance=0, member_local_indices=(0,))
        hit2 = ParameterHit(value=2.0, fragment=frag, match_instance=1, member_local_indices=(0,))
        record1 = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit1,))
        record2 = ParameterRecord(global_indices=(1,), parameter=AtomParameter.CHARGE, hits=(hit2,))
        from polymer_md.parameterisation.fragments.library import FragmentLibrary as FL
        members = FL._build_annotated_members([record1, record2])
        assert len(members) == 1


class TestResolutionStrategies:
    def test_mean_strategy_averages(self):
        strategy = MeanStrategy()
        assert pytest.approx(strategy.resolve([1.0, 3.0]), abs=1e-9) == 2.0

    def test_mean_strategy_single_value(self):
        strategy = MeanStrategy()
        assert pytest.approx(strategy.resolve([5.0]), abs=1e-9) == 5.0

    def test_first_strategy_returns_first(self):
        strategy = FirstStrategy()
        assert pytest.approx(strategy.resolve([10.0, 20.0, 30.0]), abs=1e-9) == 10.0

    def test_max_strategy_returns_maximum(self):
        strategy = MaxStrategy()
        assert pytest.approx(strategy.resolve([1.0, 5.0, 3.0]), abs=1e-9) == 5.0


# ---------------------------------------------------------------------------
# Tests: ParameterRecord.resolve
# ---------------------------------------------------------------------------

class TestParameterRecordResolve:
    def test_resolve_uses_strategy(self):
        frag = Fragment(pattern=_PATTERN)
        hit1 = ParameterHit(value=2.0, fragment=frag, match_instance=0, member_local_indices=(0,))
        hit2 = ParameterHit(value=4.0, fragment=frag, match_instance=1, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit1, hit2))
        result = record.resolve(MeanStrategy())
        assert pytest.approx(result, abs=1e-9) == 3.0

    def test_resolve_raises_with_no_hits(self):
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=())
        with pytest.raises(NoMatchError):
            record.resolve(MeanStrategy())

    def test_resolve_with_first_strategy(self):
        frag = Fragment(pattern=_PATTERN)
        hit1 = ParameterHit(value=1.5, fragment=frag, match_instance=0, member_local_indices=(0,))
        hit2 = ParameterHit(value=9.9, fragment=frag, match_instance=1, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit1, hit2))
        assert pytest.approx(record.resolve(FirstStrategy()), abs=1e-9) == 1.5


# ---------------------------------------------------------------------------
# Tests: schema versioning and multi-term dihedral round-trip
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_saved_file_contains_schema_version(self):
        library = FragmentLibrary(records=(), atom_metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            data = json.loads(path.read_text())
        assert data["schema_version"] == "2"

    def test_load_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            path.write_text(json.dumps({"schema_version": "99", "records": [], "atom_metadata": {}}))
            with pytest.raises(ValueError, match="Unsupported"):
                FragmentLibrary.load(path)

    def test_load_rejects_v1_without_schema_version_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            path.write_text(json.dumps({"records": [], "atom_metadata": {}}))
            with pytest.raises(ValueError, match="Unsupported"):
                FragmentLibrary.load(path)


class TestMultiTermDihedralRoundTrip:
    def test_multi_term_dihedral_survives_save_load(self):
        frag = Fragment(pattern=_PATTERN2)
        terms = (
            DihedralTerm(force_constant=1.0, phase=0.0, periodicity=1.0),
            DihedralTerm(force_constant=0.5, phase=3.14, periodicity=3.0),
        )
        hit = ParameterHit(value=terms, fragment=frag, match_instance=0, member_local_indices=(0, 1, 2, 3))
        record = ParameterRecord(global_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT, hits=(hit,))
        library = FragmentLibrary(records=(record,), atom_metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            loaded = FragmentLibrary.load(path)
        dihedral_records = [r for r in loaded.records if r.parameter == DihedralParameter.FORCE_CONSTANT]
        assert len(dihedral_records) == 1
        loaded_value = dihedral_records[0].hits[0].value
        assert isinstance(loaded_value, tuple)
        assert len(loaded_value) == 2
        assert pytest.approx(loaded_value[0].force_constant, abs=1e-9) == 1.0
        assert pytest.approx(loaded_value[1].force_constant, abs=1e-9) == 0.5
        assert pytest.approx(loaded_value[1].phase, abs=1e-3) == 3.14

    def test_serialised_dihedral_uses_terms_key(self):
        frag = Fragment(pattern=_PATTERN2)
        terms = (DihedralTerm(force_constant=2.0, phase=0.0, periodicity=2.0),)
        hit = ParameterHit(value=terms, fragment=frag, match_instance=0, member_local_indices=(0, 1, 2, 3))
        record = ParameterRecord(global_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT, hits=(hit,))
        library = FragmentLibrary(records=(record,), atom_metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            data = json.loads(path.read_text())
        hit_data = data["records"][0]["hits"][0]
        assert "terms" in hit_data["value"]
        assert hit_data["value"]["terms"] == [[2.0, 0.0, 2.0]]


class TestMigrateV1ToV2:
    def test_removes_dihedral_records(self):
        v1_data = {
            "records": [
                {"global_indices": [0, 1], "parameter_kind": "bond", "parameter_name": "FORCE_CONSTANT", "hits": []},
                {"global_indices": [0, 1, 2, 3], "parameter_kind": "dihedral", "parameter_name": "FORCE_CONSTANT", "hits": []},
                {"global_indices": [0, 1, 2, 3], "parameter_kind": "dihedral", "parameter_name": "PHASE", "hits": []},
            ],
            "atom_metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            path.write_text(json.dumps(v1_data))
            migrate_v1_to_v2(path)
            loaded = FragmentLibrary.load(path)
        dihedral_records = [r for r in loaded.records if isinstance(r.parameter, DihedralParameter)]
        assert len(dihedral_records) == 0
        bond_records = [r for r in loaded.records if r.parameter == BondParameter.FORCE_CONSTANT]
        assert len(bond_records) == 1

    def test_idempotent_on_v2(self):
        library = FragmentLibrary(records=(), atom_metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            library.save(path)
            migrate_v1_to_v2(path)
            loaded = FragmentLibrary.load(path)
        assert len(loaded.records) == 0
