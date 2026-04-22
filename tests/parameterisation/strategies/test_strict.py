from __future__ import annotations

import pytest

from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.strategies.base import MissingParameterError
from polymer_md.parameterisation.strategies.strict import StrictMissingParameterStrategy


class TestStrictMissingParameterStrategy:
    def test_always_raises_for_atom_parameter(self, strategy_context_s_position0):
        # arrange
        strategy = StrictMissingParameterStrategy()

        # act / assert
        with pytest.raises(MissingParameterError):
            strategy.resolve(
                global_indices=(0,),
                parameter=AtomParameter.CHARGE,
                context=strategy_context_s_position0,
            )

    def test_always_raises_for_bond_parameter(self, strategy_context_s_position0):
        # arrange
        strategy = StrictMissingParameterStrategy()

        # act / assert
        with pytest.raises(MissingParameterError):
            strategy.resolve(
                global_indices=(0, 1),
                parameter=BondParameter.FORCE_CONSTANT,
                context=strategy_context_s_position0,
            )

    def test_error_message_contains_global_indices(self, strategy_context_s_position0):
        # arrange
        strategy = StrictMissingParameterStrategy()

        # act
        with pytest.raises(MissingParameterError) as exc_info:
            strategy.resolve(
                global_indices=(42,),
                parameter=AtomParameter.CHARGE,
                context=strategy_context_s_position0,
            )

        # assert
        assert "42" in str(exc_info.value)

    def test_error_message_contains_parameter_name(self, strategy_context_s_position0):
        # arrange
        strategy = StrictMissingParameterStrategy()

        # act
        with pytest.raises(MissingParameterError) as exc_info:
            strategy.resolve(
                global_indices=(0,),
                parameter=AtomParameter.CHARGE,
                context=strategy_context_s_position0,
            )

        # assert
        assert "CHARGE" in str(exc_info.value)
