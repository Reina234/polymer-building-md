from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from polymer_md.building.data_models.constraints import TransitionConstraint
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.transition_matrix import TransitionMatrix


class TransitionMatrixSolver(ABC):
    @abstractmethod
    def solve(
        self,
        specs: list[MonomerSpec],
        constraints: Optional[list[TransitionConstraint]] = None,
    ) -> TransitionMatrix: ...
