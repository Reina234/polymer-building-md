from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rdkit import Chem

from polymer_md.parameterisation.fragments.data_models.parameters import DihedralTerm, ForceFieldParameter
from polymer_md.parameterisation.fragments.library import FragmentLibrary


class MissingParameterError(Exception):
    pass


@dataclass(frozen=True)
class StrategyContext:
    library: FragmentLibrary
    derived_mol: Chem.Mol = field(compare=False, hash=False)
    polymer_atom_metadata: dict[int, tuple[str, int]] = field(default_factory=dict)
    # polymer global_idx → (residue_id, within_residue_position)


class MissingParameterStrategy(Protocol):
    def resolve(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> float | tuple[DihedralTerm, ...]: ...
