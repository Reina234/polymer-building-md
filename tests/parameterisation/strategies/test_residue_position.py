from __future__ import annotations

import pytest

from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy


class TestResiduePositionStrategy:
    def test_averages_across_matching_trimer_contexts(self, strategy_context_s_position0):
        # arrange – library has -0.10 (from _PATTERN_CC) and -0.11 (from _PATTERN_CCC)
        # for residue S, position 0, CHARGE
        strategy = ResiduePositionStrategy(min_matches=1)

        # act
        result = strategy.resolve(
            global_indices=(0,),
            parameter=AtomParameter.CHARGE,
            context=strategy_context_s_position0,
        )

        # assert
        assert pytest.approx(result, abs=1e-6) == (-0.10 + -0.11) / 2

    def test_averages_different_position(self, strategy_context_s_position1):
        # arrange – library has -0.12 and -0.13 for residue S, position 1
        strategy = ResiduePositionStrategy(min_matches=1)

        # act
        result = strategy.resolve(
            global_indices=(1,),
            parameter=AtomParameter.CHARGE,
            context=strategy_context_s_position1,
        )

        # assert
        assert pytest.approx(result, abs=1e-6) == (-0.12 + -0.13) / 2

    def test_raises_when_below_min_matches(
        self, library_with_metadata, propane_mol
    ):
        # arrange – residue T, position 0 has only 1 match (+0.05)
        strategy = ResiduePositionStrategy(min_matches=2)
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={2: ("T", 0)},
        )

        # act / assert
        with pytest.raises(MissingParameterError, match="need at least 2"):
            strategy.resolve(
                global_indices=(2,),
                parameter=AtomParameter.CHARGE,
                context=context,
            )

    def test_raises_when_residue_not_in_library(
        self, library_with_metadata, propane_mol
    ):
        # arrange
        strategy = ResiduePositionStrategy(min_matches=1)
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={0: ("UNKNOWN_RESIDUE", 0)},
        )

        # act / assert
        with pytest.raises(MissingParameterError):
            strategy.resolve(
                global_indices=(0,),
                parameter=AtomParameter.CHARGE,
                context=context,
            )

    def test_raises_when_no_polymer_metadata(
        self, library_with_metadata, propane_mol
    ):
        # arrange
        strategy = ResiduePositionStrategy(min_matches=1)
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={},  # empty – atom 0 has no metadata
        )

        # act / assert
        with pytest.raises(MissingParameterError, match="no residue position metadata"):
            strategy.resolve(
                global_indices=(0,),
                parameter=AtomParameter.CHARGE,
                context=context,
            )

    def test_raises_for_non_atom_parameter(self, strategy_context_s_position0):
        # arrange
        strategy = ResiduePositionStrategy(min_matches=1)

        # act / assert
        with pytest.raises(MissingParameterError, match="AtomParameter"):
            strategy.resolve(
                global_indices=(0, 1),
                parameter=BondParameter.FORCE_CONSTANT,
                context=strategy_context_s_position0,
            )

    def test_single_match_satisfies_min_matches_one(
        self, library_with_metadata, propane_mol
    ):
        # arrange – residue T, position 0 has exactly 1 match
        strategy = ResiduePositionStrategy(min_matches=1)
        context = StrategyContext(
            library=library_with_metadata,
            derived_mol=propane_mol,
            polymer_atom_metadata={0: ("T", 0)},
        )

        # act
        result = strategy.resolve(
            global_indices=(0,),
            parameter=AtomParameter.CHARGE,
            context=context,
        )

        # assert
        assert pytest.approx(result, abs=1e-6) == 0.05
