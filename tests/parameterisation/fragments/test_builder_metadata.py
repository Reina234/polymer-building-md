from __future__ import annotations

import pytest

from polymer_md.parameterisation.fragments.extraction.builder import FragmentLibraryBuilder


class TestRegionPositionEntries:
    def test_homopolymer_all_regions_get_metadata(self):
        # arrange - SSS homopolymer: three distinct regions all with residue_id "S"
        regions = [
            ("S", frozenset({0, 1})),
            ("S", frozenset({2, 3})),
            ("S", frozenset({4, 5})),
        ]
        mol3d_to_parmed = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}

        # act
        result = FragmentLibraryBuilder._region_position_entries(regions, mol3d_to_parmed)

        # assert - all 6 atoms must have metadata, not just the last region's 2
        assert len(result) == 6
        assert result[10] == ("S", 0)
        assert result[11] == ("S", 1)
        assert result[12] == ("S", 0)
        assert result[13] == ("S", 1)
        assert result[14] == ("S", 0)
        assert result[15] == ("S", 1)

    def test_heteropolymer_different_residue_ids(self):
        # arrange - ABB: left is "A", central and right are "B"
        regions = [
            ("A", frozenset({0, 1})),
            ("B", frozenset({2, 3})),
            ("B", frozenset({4, 5})),
        ]
        mol3d_to_parmed = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}

        # act
        result = FragmentLibraryBuilder._region_position_entries(regions, mol3d_to_parmed)

        # assert
        assert len(result) == 6
        assert result[10] == ("A", 0)
        assert result[11] == ("A", 1)
        assert result[12] == ("B", 0)
        assert result[14] == ("B", 0)

    def test_positions_are_rank_of_sorted_mol3d_indices(self):
        # arrange
        regions = [("S", frozenset({5, 2, 8}))]
        mol3d_to_parmed = {2: 20, 5: 50, 8: 80}

        # act
        result = FragmentLibraryBuilder._region_position_entries(regions, mol3d_to_parmed)

        # assert - position is rank within sorted mol3d indices
        assert result[20] == ("S", 0)
        assert result[50] == ("S", 1)
        assert result[80] == ("S", 2)

    def test_missing_mol3d_in_crosswalk_is_skipped(self):
        # arrange
        regions = [("S", frozenset({0, 1, 2}))]
        mol3d_to_parmed = {0: 10, 2: 20}

        # act
        result = FragmentLibraryBuilder._region_position_entries(regions, mol3d_to_parmed)

        # assert - mol3d index 1 has no parmed mapping and is silently skipped
        assert len(result) == 2
        assert 10 in result
        assert 20 in result
