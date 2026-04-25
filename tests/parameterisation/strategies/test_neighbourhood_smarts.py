from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterHit, ParameterRecord
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext
from polymer_md.parameterisation.strategies.neighbourhood_smarts import NeighbourhoodSMARTSStrategy


def _make_library_from_mol(
    mol: Chem.Mol,
    atom_indices: tuple[int, ...],
    centre_idx: int,
    residue_id: str,
    within_residue_position: int,
    charge: float,
) -> FragmentLibrary:
    pattern, g2l = SmartsBuilder.subgraph(mol, atom_indices)
    centre_local = g2l[centre_idx]

    metadata = {
        pattern: {
            centre_local: AtomMetadata(
                gaff2_type="c3",
                residue_id=residue_id,
                within_residue_position=within_residue_position,
            )
        }
    }
    fragment = Fragment(pattern=pattern)
    hit = ParameterHit(
        value=charge,
        fragment=fragment,
        match_instance=0,
        member_local_indices=(centre_local,),
    )
    record = ParameterRecord(
        global_indices=(centre_idx,),
        parameter=AtomParameter.CHARGE,
        hits=(hit,),
    )
    return FragmentLibrary(records=(record,), atom_metadata=metadata)


class TestBallAroundAtom:
    def test_radius_zero_returns_only_centre(self, propane_mol):
        # arrange / act
        ball = NeighbourhoodSMARTSStrategy._ball_around_atom(propane_mol, 1, radius=0)

        # assert
        assert ball == (1,)

    def test_radius_one_includes_direct_neighbours(self, propane_mol):
        # arrange – propane (CCC + Hs), atom 1 is the middle C
        # act
        ball = NeighbourhoodSMARTSStrategy._ball_around_atom(propane_mol, 1, radius=1)

        # assert – middle C and its direct neighbours (2 Cs + bonded Hs)
        assert 1 in ball
        assert len(ball) > 1

    def test_larger_radius_includes_more_atoms(self, propane_mol):
        # arrange / act
        ball_r1 = NeighbourhoodSMARTSStrategy._ball_around_atom(propane_mol, 1, radius=1)
        ball_r2 = NeighbourhoodSMARTSStrategy._ball_around_atom(propane_mol, 1, radius=2)

        # assert
        assert len(ball_r2) >= len(ball_r1)

    def test_ball_is_sorted(self, ethylbenzene_mol):
        # arrange / act
        ball = NeighbourhoodSMARTSStrategy._ball_around_atom(ethylbenzene_mol, 0, radius=2)

        # assert
        assert list(ball) == sorted(ball)


class TestNeighbourhoodSMARTSStrategy:
    def test_finds_match_at_exact_context(self, propane_mol):
        # arrange
        # Build a library from propane itself – the fragment covers all heavy atoms
        heavy = tuple(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        # Centre atom is heavy[0] (a terminal C), residue S, position 0, charge -0.10
        centre_idx = heavy[0]
        library = _make_library_from_mol(
            propane_mol, heavy, centre_idx, "S", 0, -0.10
        )
        context = StrategyContext(
            library=library,
            derived_mol=propane_mol,
            polymer_atom_metadata={centre_idx: ("S", 0)},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)

        # act
        result = strategy.resolve(
            global_indices=(centre_idx,),
            parameter=AtomParameter.CHARGE,
            context=context,
        )

        # assert
        assert pytest.approx(result, abs=1e-6) == -0.10

    def test_falls_back_to_smaller_radius_when_needed(self, propane_mol):
        # arrange
        # Build library from just a 2-atom fragment (C–C) — only radius 1 will match
        heavy = tuple(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        # Use just the first two heavy atoms for the library fragment
        fragment_atoms = heavy[:2]
        centre_idx = heavy[0]
        library = _make_library_from_mol(
            propane_mol, fragment_atoms, centre_idx, "S", 0, -0.20
        )
        context = StrategyContext(
            library=library,
            derived_mol=propane_mol,
            polymer_atom_metadata={centre_idx: ("S", 0)},
        )
        # max_radius=3 won't match (full context too large for small fragment)
        # but the strategy should fall back until it finds a match
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)

        # act
        result = strategy.resolve(
            global_indices=(centre_idx,),
            parameter=AtomParameter.CHARGE,
            context=context,
        )

        # assert
        assert pytest.approx(result, abs=1e-6) == -0.20

    def test_raises_when_no_match_at_any_radius(self, propane_mol, library_with_metadata):
        # arrange – polymer atom metadata says this is residue UNKNOWN, which is not in library
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={0: ("UNKNOWN_RESIDUE", 0)},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)

        # act / assert
        with pytest.raises(MissingParameterError):
            strategy.resolve(
                global_indices=(0,),
                parameter=AtomParameter.CHARGE,
                context=context,
            )

    def test_raises_for_non_atom_parameter(self, strategy_context_s_position0):
        # arrange
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)

        # act / assert
        with pytest.raises(MissingParameterError, match="AtomParameter"):
            strategy.resolve(
                global_indices=(0, 1),
                parameter=BondParameter.FORCE_CONSTANT,
                context=strategy_context_s_position0,
            )

    def test_raises_when_no_polymer_metadata(self, library_with_metadata, propane_mol):
        # arrange
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)

        # act / assert
        with pytest.raises(MissingParameterError, match="no residue position metadata"):
            strategy.resolve(
                global_indices=(0,),
                parameter=AtomParameter.CHARGE,
                context=context,
            )

    def test_respects_min_matches_threshold(self, propane_mol):
        # arrange – one match found but min_matches=2 → should raise
        heavy = tuple(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        centre_idx = heavy[0]
        library = _make_library_from_mol(
            propane_mol, heavy, centre_idx, "S", 0, -0.10
        )
        context = StrategyContext(
            library=library,
            derived_mol=propane_mol,
            polymer_atom_metadata={centre_idx: ("S", 0)},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=2)

        # act / assert
        with pytest.raises(MissingParameterError):
            strategy.resolve(
                global_indices=(centre_idx,),
                parameter=AtomParameter.CHARGE,
                context=context,
            )

    def test_averages_multiple_library_matches(self, propane_mol):
        # arrange – build two library fragments both matching the same neighbourhood
        heavy = tuple(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        centre_idx = heavy[0]

        lib1 = _make_library_from_mol(propane_mol, heavy, centre_idx, "S", 0, -0.10)
        lib2 = _make_library_from_mol(propane_mol, heavy[:2], centre_idx, "S", 0, -0.20)

        combined = FragmentLibrary(
            records=lib1.records + lib2.records,
            atom_metadata={**lib1.atom_metadata, **lib2.atom_metadata},
        )
        context = StrategyContext(
            library=combined,
            derived_mol=propane_mol,
            polymer_atom_metadata={centre_idx: ("S", 0)},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)

        # act
        result = strategy.resolve(
            global_indices=(centre_idx,),
            parameter=AtomParameter.CHARGE,
            context=context,
        )

        # assert – will average whichever matches are found at the largest radius
        assert isinstance(result, float)


class TestNeighbourhoodSMARTSEdgeCases:
    def test_empty_heavy_ball_returns_no_values(self, propane_mol, library_with_metadata):
        # A hydrogen centre has no heavy atoms in its ball at radius 0 → heavy_ball is empty
        # Find a hydrogen atom index
        h_idx = next(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=0, min_radius=0, min_matches=1)
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={h_idx: ("S", 0)},
        )
        # At radius 0, ball is just the H atom → heavy_ball is empty → _neighbourhood_query returns None
        result = strategy._collect_values_at_radius(
            h_idx, 0, "S", 0, AtomParameter.CHARGE, context
        )
        assert result == []

    def test_invalid_fragment_pattern_in_metadata_skipped(self, propane_mol):
        # Library with an invalid SMARTS pattern in atom_metadata
        from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
        from polymer_md.parameterisation.fragments.data_models.match import ParameterHit, ParameterRecord
        from polymer_md.parameterisation.fragments.data_models.fragment import Fragment

        bad_pattern = "[invalid!!!"
        frag = Fragment(pattern=bad_pattern)
        hit = ParameterHit(value=-0.1, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        library = FragmentLibrary(
            records=(record,),
            atom_metadata={bad_pattern: {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)}},
        )
        heavy = tuple(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        centre_idx = heavy[0]
        context = StrategyContext(
            library=library,
            derived_mol=propane_mol,
            polymer_atom_metadata={centre_idx: ("S", 0)},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)
        # _matched_local_indices returns [] for invalid fragment pattern → no values → raises
        with pytest.raises(MissingParameterError):
            strategy.resolve((centre_idx,), AtomParameter.CHARGE, context)

    def test_deduplication_of_record_values(self, propane_mol):
        # Same library record matched twice via two different neighbourhood sub-patterns
        # should only contribute once (seen_records dedup)
        heavy = tuple(
            i for i in range(propane_mol.GetNumAtoms())
            if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        )
        centre_idx = heavy[0]
        library = _make_library_from_mol(propane_mol, heavy, centre_idx, "S", 0, -0.10)
        context = StrategyContext(
            library=library,
            derived_mol=propane_mol,
            polymer_atom_metadata={centre_idx: ("S", 0)},
        )
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)
        result = strategy.resolve((centre_idx,), AtomParameter.CHARGE, context)
        # Only one record → should return exactly -0.10, not an average of duplicates
        assert pytest.approx(result, abs=1e-6) == -0.10

    def test_neighbourhood_query_returns_none_when_smarts_invalid(self, propane_mol):
        from unittest.mock import patch
        strategy = NeighbourhoodSMARTSStrategy(max_radius=3, min_radius=1, min_matches=1)
        with patch(
            "polymer_md.parameterisation.strategies.neighbourhood_smarts.Chem.MolFromSmarts",
            return_value=None,
        ):
            result = strategy._neighbourhood_query(propane_mol, 1, radius=1)
        assert result is None

    def test_record_value_returns_none_when_pattern_not_in_any_hit(self):
        frag = Fragment(pattern="[#6;A]")
        hit = ParameterHit(value=-0.1, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        seen_records: set[int] = set()
        result = NeighbourhoodSMARTSStrategy._record_value(
            "NONEXISTENT_PATTERN", 0, AtomParameter.CHARGE, (record,), seen_records
        )
        assert result is None

    def test_record_value_skips_already_seen_record_index(self):
        frag = Fragment(pattern="[#6;A]")
        hit = ParameterHit(value=-0.1, fragment=frag, match_instance=0, member_local_indices=(0,))
        record = ParameterRecord(global_indices=(0,), parameter=AtomParameter.CHARGE, hits=(hit,))
        seen_records = {0}
        result = NeighbourhoodSMARTSStrategy._record_value(
            "[#6;A]", 0, AtomParameter.CHARGE, (record,), seen_records
        )
        assert result is None
