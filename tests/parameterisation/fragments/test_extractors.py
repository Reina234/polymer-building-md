from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from polymer_md.building.data_models.trimer import BondType, TrimerResult
from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.extraction.interior import InteriorFragmentExtractor
from polymer_md.parameterisation.fragments.extraction.region_extractor import RegionFragmentExtractor
from polymer_md.parameterisation.fragments.extraction.terminal import TerminalFragmentExtractor
from polymer_md.utils.parmed_helper import CoordinateCrosswalk, StructureMolDeriver


_LEFT = frozenset({0, 1})
_CENTRAL = frozenset({2, 3})
_RIGHT = frozenset({4, 5})


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


class TestInteriorFragmentExtractor:
    def test_extract_returns_one_pair(self):
        pt = _make_parameterised_trimer()
        mock_pair = (MagicMock(spec=Fragment), {0: 0})
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=mock_pair),
        ):
            result = InteriorFragmentExtractor().extract(pt)
        assert len(result) == 1

    def test_extract_result_is_the_region_extractor_output(self):
        pt = _make_parameterised_trimer()
        expected_pair = (MagicMock(spec=Fragment), {0: 0, 1: 1})
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=expected_pair),
        ):
            result = InteriorFragmentExtractor().extract(pt)
        assert result[0] is expected_pair

    def test_extract_context_is_union_of_all_three_regions(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=(MagicMock(), {})) as mock_extract,
        ):
            InteriorFragmentExtractor().extract(pt)
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["context_parmed_indices"] == _LEFT | _CENTRAL | _RIGHT

    def test_extract_region_is_central_not_left_or_right(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=(MagicMock(), {})) as mock_extract,
        ):
            InteriorFragmentExtractor().extract(pt)
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["region_parmed_indices"] == _CENTRAL
        assert kwargs["region_parmed_indices"] != _LEFT
        assert kwargs["region_parmed_indices"] != _RIGHT

    def test_extract_calls_resolve_parmed_indices_three_times(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(
                RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]
            ) as mock_resolve,
            patch.object(RegionFragmentExtractor, "extract", return_value=(MagicMock(), {})),
        ):
            InteriorFragmentExtractor().extract(pt)
        assert mock_resolve.call_count == 3

    def test_extract_calls_region_extract_exactly_once(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=(MagicMock(), {})) as mock_extract,
        ):
            InteriorFragmentExtractor().extract(pt)
        assert mock_extract.call_count == 1

    def test_extract_derives_mol_from_structure(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()) as mock_derive,
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=(MagicMock(), {})),
        ):
            InteriorFragmentExtractor().extract(pt)
        mock_derive.assert_called_once_with(pt.structure)

    def test_extract_passes_derived_mol_to_region_extract(self):
        pt = _make_parameterised_trimer()
        mock_derived = MagicMock()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=mock_derived),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(RegionFragmentExtractor, "extract", return_value=(MagicMock(), {})) as mock_extract,
        ):
            InteriorFragmentExtractor().extract(pt)
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["derived_mol"] is mock_derived


class TestTerminalFragmentExtractor:
    def test_extract_returns_two_pairs(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ),
        ):
            result = TerminalFragmentExtractor().extract(pt)
        assert len(result) == 2

    def test_extract_first_pair_region_is_left(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ) as mock_extract,
        ):
            TerminalFragmentExtractor().extract(pt)
        first_kwargs = mock_extract.call_args_list[0].kwargs
        assert first_kwargs["region_parmed_indices"] == _LEFT

    def test_extract_second_pair_region_is_right(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ) as mock_extract,
        ):
            TerminalFragmentExtractor().extract(pt)
        second_kwargs = mock_extract.call_args_list[1].kwargs
        assert second_kwargs["region_parmed_indices"] == _RIGHT

    def test_extract_both_calls_share_same_context(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ) as mock_extract,
        ):
            TerminalFragmentExtractor().extract(pt)
        first_ctx = mock_extract.call_args_list[0].kwargs["context_parmed_indices"]
        second_ctx = mock_extract.call_args_list[1].kwargs["context_parmed_indices"]
        assert first_ctx == second_ctx == _LEFT | _CENTRAL | _RIGHT

    def test_extract_calls_resolve_parmed_indices_three_times(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(
                RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]
            ) as mock_resolve,
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ),
        ):
            TerminalFragmentExtractor().extract(pt)
        assert mock_resolve.call_count == 3

    def test_extract_calls_region_extract_exactly_twice(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ) as mock_extract,
        ):
            TerminalFragmentExtractor().extract(pt)
        assert mock_extract.call_count == 2

    def test_extract_neither_pair_region_is_central(self):
        pt = _make_parameterised_trimer()
        with (
            patch.object(StructureMolDeriver, "derive", return_value=MagicMock()),
            patch.object(CoordinateCrosswalk, "map_mol3d_to_parmed", return_value={}),
            patch.object(RegionFragmentExtractor, "resolve_parmed_indices", side_effect=[_LEFT, _CENTRAL, _RIGHT]),
            patch.object(
                RegionFragmentExtractor, "extract", side_effect=[(MagicMock(), {}), (MagicMock(), {})]
            ) as mock_extract,
        ):
            TerminalFragmentExtractor().extract(pt)
        regions = [call.kwargs["region_parmed_indices"] for call in mock_extract.call_args_list]
        assert _CENTRAL not in regions
