from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from polymer_md.building.data_models.trimer import BondType, TrimerResult
from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterRecord
from polymer_md.parameterisation.fragments.extraction.builder import FragmentLibraryBuilder
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.utils.parmed_helper import CoordinateCrosswalk, StructureMolDeriver


def _make_parameterised_trimer() -> ParameterisedTrimer:
    mol = Chem.MolFromSmiles("CCC")
    trimer_result = TrimerResult(
        left_id="S",
        central_id="S",
        right_id="S",
        left_bond=BondType(from_site=0, to_site=1),
        right_bond=BondType(from_site=1, to_site=0),
        mol=mol,
        probability=1.0,
        left_atom_indices=frozenset({0, 1}),
        central_atom_indices=frozenset({2, 3}),
        right_atom_indices=frozenset({4, 5}),
        cap_atom_indices=frozenset(),
    )
    return ParameterisedTrimer(
        trimer_result=trimer_result,
        gromacs_files=MagicMock(),
        structure=MagicMock(),
        mol_3d=MagicMock(),
    )


# ---------------------------------------------------------------------------
# FragmentLibraryBuilder.build
# ---------------------------------------------------------------------------

class TestFragmentLibraryBuilderBuild:
    def test_build_returns_fragment_library(self):
        pt = _make_parameterised_trimer()
        with patch.object(FragmentLibraryBuilder, "_process_trimer", return_value=([], {})):
            result = FragmentLibraryBuilder().build([pt])
        assert isinstance(result, FragmentLibrary)

    def test_build_accumulates_records_from_all_trimers(self):
        pt1 = _make_parameterised_trimer()
        pt2 = _make_parameterised_trimer()
        records1 = [MagicMock(spec=ParameterRecord)]
        records2 = [MagicMock(spec=ParameterRecord), MagicMock(spec=ParameterRecord)]
        with patch.object(
            FragmentLibraryBuilder,
            "_process_trimer",
            side_effect=[(records1, {}), (records2, {})],
        ):
            result = FragmentLibraryBuilder().build([pt1, pt2])
        assert len(result.records) == 3

    def test_build_empty_input_returns_empty_library(self):
        result = FragmentLibraryBuilder().build([])
        assert isinstance(result, FragmentLibrary)
        assert len(result.records) == 0
        assert result.atom_metadata == {}

    def test_build_merges_metadata_from_all_trimers(self):
        pt1 = _make_parameterised_trimer()
        pt2 = _make_parameterised_trimer()
        meta1 = {"pattern_a": {0: MagicMock(spec=AtomMetadata)}}
        meta2 = {"pattern_b": {0: MagicMock(spec=AtomMetadata)}}
        with patch.object(
            FragmentLibraryBuilder,
            "_process_trimer",
            side_effect=[([], meta1), ([], meta2)],
        ):
            result = FragmentLibraryBuilder().build([pt1, pt2])
        assert "pattern_a" in result.atom_metadata
        assert "pattern_b" in result.atom_metadata

    def test_build_calls_process_trimer_once_per_input(self):
        pts = [_make_parameterised_trimer() for _ in range(3)]
        with patch.object(
            FragmentLibraryBuilder, "_process_trimer", return_value=([], {})
        ) as mock_process:
            FragmentLibraryBuilder().build(pts)
        assert mock_process.call_count == 3


# ---------------------------------------------------------------------------
# FragmentLibraryBuilder._extract_all_fragment_pairs
# ---------------------------------------------------------------------------

class TestExtractAllFragmentPairs:
    def test_combines_interior_and_terminal_pairs(self):
        pt = _make_parameterised_trimer()
        interior_result = [(MagicMock(spec=Fragment), {0: 0})]
        terminal_result = [
            (MagicMock(spec=Fragment), {1: 1}),
            (MagicMock(spec=Fragment), {2: 2}),
        ]
        mock_interior = MagicMock()
        mock_interior.extract.return_value = interior_result
        mock_terminal = MagicMock()
        mock_terminal.extract.return_value = terminal_result

        builder = FragmentLibraryBuilder(
            interior_extractor=mock_interior,
            terminal_extractor=mock_terminal,
        )
        result = builder._extract_all_fragment_pairs(pt)
        assert len(result) == 3
        assert result == interior_result + terminal_result

    def test_interior_extractor_called_with_parameterised_trimer(self):
        pt = _make_parameterised_trimer()
        mock_interior = MagicMock()
        mock_interior.extract.return_value = []
        mock_terminal = MagicMock()
        mock_terminal.extract.return_value = []

        FragmentLibraryBuilder(
            interior_extractor=mock_interior,
            terminal_extractor=mock_terminal,
        )._extract_all_fragment_pairs(pt)

        mock_interior.extract.assert_called_once_with(pt)
        mock_terminal.extract.assert_called_once_with(pt)


# ---------------------------------------------------------------------------
# FragmentLibraryBuilder._process_trimer
# ---------------------------------------------------------------------------

class TestProcessTrimer:
    def test_process_trimer_returns_records_and_metadata(self):
        pt = _make_parameterised_trimer()
        mock_fragment = MagicMock(spec=Fragment)
        mock_fragment.pattern = "[#6;A]"
        fragment_pairs = [(mock_fragment, {0: 0})]
        mock_records = [MagicMock(spec=ParameterRecord)]
        mock_meta = {"pattern": {}}

        with (
            patch.object(FragmentLibraryBuilder, "_extract_all_fragment_pairs", return_value=fragment_pairs),
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(FragmentLibraryBuilder, "_build_atom_metadata", return_value=mock_meta),
            patch("polymer_md.parameterisation.fragments.extraction.builder.FragmentMatcher") as MockMatcher,
        ):
            MockMatcher.return_value.match_all.return_value = []
            MockMatcher.return_value.build_records.return_value = mock_records
            records, metadata = FragmentLibraryBuilder()._process_trimer(pt)

        assert records == mock_records
        assert metadata == mock_meta

    def test_process_trimer_passes_fragments_to_matcher(self):
        pt = _make_parameterised_trimer()
        mock_fragment = MagicMock(spec=Fragment)
        mock_fragment.pattern = "[#6;A]"
        fragment_pairs = [(mock_fragment, {0: 0})]
        mock_derived = MagicMock()

        with (
            patch.object(FragmentLibraryBuilder, "_extract_all_fragment_pairs", return_value=fragment_pairs),
            patch.object(StructureMolDeriver, "derive", return_value=mock_derived),
            patch.object(FragmentLibraryBuilder, "_build_atom_metadata", return_value={}),
            patch("polymer_md.parameterisation.fragments.extraction.builder.FragmentMatcher") as MockMatcher,
        ):
            MockMatcher.return_value.match_all.return_value = []
            MockMatcher.return_value.build_records.return_value = []
            FragmentLibraryBuilder()._process_trimer(pt)

        MockMatcher.assert_called_once_with([mock_fragment])
        MockMatcher.return_value.match_all.assert_called_once_with(mock_derived)

    def test_process_trimer_derives_mol_from_structure(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(FragmentLibraryBuilder, "_extract_all_fragment_pairs", return_value=[]),
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()) as mock_derive,
            patch.object(FragmentLibraryBuilder, "_build_atom_metadata", return_value={}),
            patch("polymer_md.parameterisation.fragments.extraction.builder.FragmentMatcher") as MockMatcher,
        ):
            MockMatcher.return_value.match_all.return_value = []
            MockMatcher.return_value.build_records.return_value = []
            FragmentLibraryBuilder()._process_trimer(pt)
        mock_derive.assert_called_once_with(pt.structure)


# ---------------------------------------------------------------------------
# FragmentLibraryBuilder._build_region_position_maps
# ---------------------------------------------------------------------------

class TestBuildRegionPositionMaps:
    def test_maps_all_three_regions(self):
        pt = _make_parameterised_trimer()
        mol3d_to_parmed = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}

        result = FragmentLibraryBuilder._build_region_position_maps(pt, mol3d_to_parmed)

        assert result[10] == ("S", 0)
        assert result[11] == ("S", 1)
        assert result[12] == ("S", 0)
        assert result[13] == ("S", 1)
        assert result[14] == ("S", 0)
        assert result[15] == ("S", 1)

    def test_total_entry_count_equals_number_of_mapped_atoms(self):
        pt = _make_parameterised_trimer()
        mol3d_to_parmed = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
        result = FragmentLibraryBuilder._build_region_position_maps(pt, mol3d_to_parmed)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# FragmentLibraryBuilder._metadata_for_pattern
# ---------------------------------------------------------------------------

class TestMetadataForPattern:
    def _make_atom(self, atomic_number: int, atom_type_name: str | None = "c3") -> MagicMock:
        atom = MagicMock()
        atom.atomic_number = atomic_number
        if atom_type_name is not None:
            atom.atom_type = MagicMock()
            atom.atom_type.name = atom_type_name
        else:
            atom.atom_type = None
        return atom

    def test_assigns_metadata_for_heavy_atom(self):
        structure = MagicMock()
        structure.atoms = {0: self._make_atom(6, "c3")}

        result = FragmentLibraryBuilder._metadata_for_pattern(
            global_to_local={0: 2},
            structure=structure,
            parmed_to_mol3d={0: 0},
            region_maps={0: ("S", 1)},
        )

        assert 2 in result
        assert result[2].gaff2_type == "c3"
        assert result[2].residue_id == "S"
        assert result[2].within_residue_position == 1

    def test_skips_hydrogen_atoms(self):
        structure = MagicMock()
        structure.atoms = {0: self._make_atom(1, "ha")}

        result = FragmentLibraryBuilder._metadata_for_pattern(
            global_to_local={0: 0},
            structure=structure,
            parmed_to_mol3d={0: 0},
            region_maps={0: ("S", 0)},
        )
        assert result == {}

    def test_skips_atom_not_in_region_maps(self):
        structure = MagicMock()
        structure.atoms = {0: self._make_atom(6)}

        result = FragmentLibraryBuilder._metadata_for_pattern(
            global_to_local={0: 0},
            structure=structure,
            parmed_to_mol3d={0: 0},
            region_maps={},
        )
        assert result == {}

    def test_gaff2_type_empty_when_atom_type_is_none(self):
        structure = MagicMock()
        structure.atoms = {0: self._make_atom(6, atom_type_name=None)}

        result = FragmentLibraryBuilder._metadata_for_pattern(
            global_to_local={0: 0},
            structure=structure,
            parmed_to_mol3d={0: 0},
            region_maps={0: ("S", 0)},
        )
        assert result[0].gaff2_type == ""

    def test_multiple_heavy_atoms_all_assigned(self):
        structure = MagicMock()
        structure.atoms = {
            0: self._make_atom(6, "c3"),
            1: self._make_atom(6, "c3"),
        }
        result = FragmentLibraryBuilder._metadata_for_pattern(
            global_to_local={0: 0, 1: 1},
            structure=structure,
            parmed_to_mol3d={0: 0, 1: 1},
            region_maps={0: ("S", 0), 1: ("S", 1)},
        )
        assert len(result) == 2
        assert result[0].within_residue_position == 0
        assert result[1].within_residue_position == 1

    def test_hydrogen_mixed_with_heavy_only_heavy_assigned(self):
        structure = MagicMock()
        structure.atoms = {
            0: self._make_atom(6, "c3"),
            1: self._make_atom(1, "ha"),
        }
        result = FragmentLibraryBuilder._metadata_for_pattern(
            global_to_local={0: 0, 1: 1},
            structure=structure,
            parmed_to_mol3d={0: 0, 1: 1},
            region_maps={0: ("S", 0), 1: ("S", 1)},
        )
        assert len(result) == 1
        assert 0 in result
        assert 1 not in result


# ---------------------------------------------------------------------------
# FragmentLibraryBuilder._build_atom_metadata
# ---------------------------------------------------------------------------

class TestBuildAtomMetadata:
    def test_deduplicates_identical_patterns(self):
        pt = _make_parameterised_trimer()
        mock_fragment = MagicMock(spec=Fragment)
        mock_fragment.pattern = "[#6;A]"
        fragment_pairs = [
            (mock_fragment, {0: 0}),
            (mock_fragment, {0: 0}),
        ]

        with (
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(FragmentLibraryBuilder, "_build_region_position_maps", return_value={}),
            patch.object(
                FragmentLibraryBuilder, "_metadata_for_pattern", return_value={}
            ) as mock_meta,
        ):
            FragmentLibraryBuilder()._build_atom_metadata(fragment_pairs, pt)

        assert mock_meta.call_count == 1

    def test_different_patterns_both_processed(self):
        pt = _make_parameterised_trimer()
        frag_a = MagicMock(spec=Fragment)
        frag_a.pattern = "pattern_a"
        frag_b = MagicMock(spec=Fragment)
        frag_b.pattern = "pattern_b"
        fragment_pairs = [(frag_a, {0: 0}), (frag_b, {1: 0})]

        with (
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(FragmentLibraryBuilder, "_build_region_position_maps", return_value={}),
            patch.object(
                FragmentLibraryBuilder, "_metadata_for_pattern", return_value={}
            ) as mock_meta,
        ):
            FragmentLibraryBuilder()._build_atom_metadata(fragment_pairs, pt)

        assert mock_meta.call_count == 2
