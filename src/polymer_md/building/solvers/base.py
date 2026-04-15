from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from polymer_md.building.data_models.composition import MonomerComposition
from polymer_md.building.data_models.constraints import TransitionConstraint
from polymer_md.building.data_models.transition_matrix import TransitionMatrix


@runtime_checkable
class TransitionMatrixSolver(Protocol):

    def solve(
        self,
        compositions: list[MonomerComposition],
        sites_per_monomer: dict[str, int],
        constraints: Optional[list[TransitionConstraint]] = None,
    ) -> TransitionMatrix: ...
