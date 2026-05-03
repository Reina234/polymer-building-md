from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from polymer_md.building.data_models.constraints import TransitionConstraint
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.transition_matrix import SiteKey, TransitionMatrix
from polymer_md.building.solvers.base import TransitionMatrixSolver


@dataclass
class ProportionalSolver(TransitionMatrixSolver):
    ht_fraction: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.ht_fraction <= 1.0:
            raise ValueError(
                f"ht_fraction must be between 0 and 1, got {self.ht_fraction}"
            )

    def solve(
        self,
        specs: list[MonomerSpec],
        constraints: Optional[list[TransitionConstraint]] = None,
    ) -> TransitionMatrix:
        if constraints:
            raise ValueError(
                "ProportionalSolver does not support constraints. Use ScipySolver."
            )

        total_weight = sum(spec.weight for spec in specs)

        site_keys: list[SiteKey] = []
        for spec in specs:
            site_keys.append(SiteKey.head(spec.residue_id))
            site_keys.append(SiteKey.tail(spec.residue_id))

        site_probability: dict[SiteKey, float] = {}
        for spec in specs:
            base = spec.weight / total_weight
            site_probability[SiteKey.head(spec.residue_id)] = base * (
                1.0 - self.ht_fraction
            )
            site_probability[SiteKey.tail(spec.residue_id)] = base * self.ht_fraction

        n_total_sites = len(site_keys)
        weights = np.array(
            [[site_probability[sk] for sk in site_keys] for _ in range(n_total_sites)],
            dtype=float,
        )

        return TransitionMatrix(site_keys=tuple(site_keys), weights=weights)
