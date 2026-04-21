from __future__ import annotations

from typing import Optional

import numpy as np

from polymer_md.building.data_models.constraints import TransitionConstraint
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.transition_matrix import SiteKey, TransitionMatrix
from polymer_md.building.solvers.base import TransitionMatrixSolver


class ProportionalSolver(TransitionMatrixSolver):
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
                1.0 - spec.ht_fraction
            )
            site_probability[SiteKey.tail(spec.residue_id)] = base * spec.ht_fraction

        n_total_sites = len(site_keys)
        weights = np.array(
            [[site_probability[sk] for sk in site_keys] for _ in range(n_total_sites)],
            dtype=float,
        )

        return TransitionMatrix(site_keys=tuple(site_keys), weights=weights)
