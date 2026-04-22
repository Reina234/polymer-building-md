from __future__ import annotations

from polymer_md.parameterisation.fragments.data_models.parameters import ForceFieldParameter
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext


class StrictMissingParameterStrategy:
    def resolve(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> float:
        raise MissingParameterError(
            f"No library match for {parameter} at polymer atom indices {global_indices}. "
            f"Library has {len(context.library.records)} records."
        )
