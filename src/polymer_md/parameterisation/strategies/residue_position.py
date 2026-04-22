from __future__ import annotations

from dataclasses import dataclass

from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, ForceFieldParameter
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext


@dataclass
class ResiduePositionStrategy:
    min_matches: int = 1

    def resolve(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> float:
        if not isinstance(parameter, AtomParameter):
            raise MissingParameterError(
                f"ResiduePositionStrategy only supports AtomParameter, got {type(parameter).__name__}"
            )

        atom_idx = global_indices[0]
        residue_id, within_residue_position = self._polymer_residue_position(
            atom_idx, context
        )
        values = context.library.hit_values_for_residue_position(
            residue_id, within_residue_position, parameter
        )

        if len(values) < self.min_matches:
            raise MissingParameterError(
                f"ResiduePositionStrategy: found {len(values)} matches for "
                f"({residue_id}, position={within_residue_position}, {parameter}), "
                f"need at least {self.min_matches}."
            )

        return sum(values) / len(values)

    @staticmethod
    def _polymer_residue_position(
        atom_idx: int,
        context: StrategyContext,
    ) -> tuple[str, int]:
        result = context.polymer_atom_metadata.get(atom_idx)
        if result is None:
            raise MissingParameterError(
                f"Atom index {atom_idx} has no residue position metadata in context. "
                f"Ensure polymer_atom_metadata is populated before calling this strategy."
            )
        return result
