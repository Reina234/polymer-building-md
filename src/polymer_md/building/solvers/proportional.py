from __future__ import annotations

from typing import Optional

import numpy as np

from polymer_md.building.data_models.composition import MonomerComposition
from polymer_md.building.data_models.constraints import TransitionConstraint
from polymer_md.building.data_models.transition_matrix import SiteKey, TransitionMatrix
from polymer_md.building.solvers.base import TransitionMatrixSolver


class ProportionalSolver(TransitionMatrixSolver):
    def solve(
        self,
        compositions: list[MonomerComposition],
        sites_per_monomer: dict[str, int],
        constraints: Optional[list[TransitionConstraint]] = None,
    ) -> TransitionMatrix:
        if constraints:
            raise ValueError(
                "ProportionalSolver does not support constraints. Use ScipySolver."
            )

        total_weight = sum(comp.weight for comp in compositions)

        site_keys: list[SiteKey] = []
        for comp in compositions:
            for site_index in range(sites_per_monomer[comp.residue_id]):
                site_keys.append(SiteKey(comp.residue_id, site_index))

        site_probability: dict[SiteKey, float] = {}
        for comp in compositions:
            num_sites = sites_per_monomer[comp.residue_id]
            prob = (comp.weight / total_weight) / num_sites
            for site_index in range(num_sites):
                site_probability[SiteKey(comp.residue_id, site_index)] = prob

        n_total_sites = len(site_keys)
        weights = np.array(
            [[site_probability[sk] for sk in site_keys] for _ in range(n_total_sites)],
            dtype=float,
        )

        return TransitionMatrix(site_keys=tuple(site_keys), weights=weights)
