from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog

from polymer_md.building.data_models.constraints import (
    FixedWeightConstraint,
    RatioConstraint,
    TransitionConstraint,
)
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.transition_matrix import SiteKey, TransitionMatrix
from polymer_md.building.solvers.base import TransitionMatrixSolver


@dataclass
class ScipySolver(TransitionMatrixSolver):
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
        constraints = constraints or []

        site_keys = self._build_site_keys(specs)
        stationary_dist = self._build_stationary_distribution(specs)
        site_to_index = {sk: i for i, sk in enumerate(site_keys)}
        n_total_sites = len(site_keys)

        equality_rows = (
            self._row_normalization_constraints(n_total_sites)
            + self._stationary_distribution_constraints(n_total_sites, stationary_dist)
            + self._user_defined_constraint_rows(
                n_total_sites, site_to_index, constraints
            )
        )
        constraint_rows = np.array([row for row, _ in equality_rows])
        constraint_rhs = np.array([rhs for _, rhs in equality_rows])

        result = linprog(
            np.zeros(n_total_sites * n_total_sites),
            A_eq=constraint_rows,
            b_eq=constraint_rhs,
            bounds=[(0.0, None)] * (n_total_sites * n_total_sites),
            method="highs",
        )

        if not result.success:
            raise ValueError(
                f"Could not solve for a valid transition matrix: {result.message}"
            )

        return TransitionMatrix(
            site_keys=tuple(site_keys),
            weights=result.x.reshape(n_total_sites, n_total_sites),
        )

    @staticmethod
    def _build_site_keys(specs: list[MonomerSpec]) -> list[SiteKey]:
        site_keys: list[SiteKey] = []
        for spec in specs:
            site_keys.append(SiteKey.head(spec.residue_id))
            site_keys.append(SiteKey.tail(spec.residue_id))
        return site_keys

    def _build_stationary_distribution(self, specs: list[MonomerSpec]) -> np.ndarray:
        total_weight = sum(spec.weight for spec in specs)
        stationary_dist: list[float] = []
        for spec in specs:
            base = spec.weight / total_weight
            stationary_dist.append(base * (1.0 - self.ht_fraction))
            stationary_dist.append(base * self.ht_fraction)
        return np.array(stationary_dist)

    @staticmethod
    def _row_normalization_constraints(
        n_total_sites: int,
    ) -> list[tuple[np.ndarray, float]]:
        constraints = []
        for row_index in range(n_total_sites):
            row = np.zeros(n_total_sites * n_total_sites)
            row[row_index * n_total_sites : (row_index + 1) * n_total_sites] = 1.0
            constraints.append((row, 1.0))
        return constraints

    @staticmethod
    def _stationary_distribution_constraints(
        n_total_sites: int,
        stationary_dist: np.ndarray,
    ) -> list[tuple[np.ndarray, float]]:
        constraints = []
        for col_index in range(n_total_sites):
            row = np.zeros(n_total_sites * n_total_sites)
            for row_index in range(n_total_sites):
                row[row_index * n_total_sites + col_index] = stationary_dist[row_index]
            constraints.append((row, float(stationary_dist[col_index])))
        return constraints

    @staticmethod
    def _fixed_weight_constraint_row(
        n_total_sites: int,
        constraint: FixedWeightConstraint,
        site_to_index: dict[SiteKey, int],
    ) -> tuple[np.ndarray, float]:
        row = np.zeros(n_total_sites * n_total_sites)
        from_idx = site_to_index[constraint.from_site]
        to_idx = site_to_index[constraint.to_site]
        row[from_idx * n_total_sites + to_idx] = 1.0
        return row, constraint.weight

    @staticmethod
    def _ratio_constraint_row(
        n_total_sites: int,
        constraint: RatioConstraint,
        site_to_index: dict[SiteKey, int],
    ) -> tuple[np.ndarray, float]:
        row = np.zeros(n_total_sites * n_total_sites)
        from_a_idx = site_to_index[constraint.from_a]
        to_a_idx = site_to_index[constraint.to_a]
        from_b_idx = site_to_index[constraint.from_b]
        to_b_idx = site_to_index[constraint.to_b]
        row[from_a_idx * n_total_sites + to_a_idx] = 1.0
        row[from_b_idx * n_total_sites + to_b_idx] = -constraint.ratio
        return row, 0.0

    def _user_defined_constraint_rows(
        self,
        n_total_sites: int,
        site_to_index: dict[SiteKey, int],
        constraints: list[TransitionConstraint],
    ) -> list[tuple[np.ndarray, float]]:
        rows = []
        for constraint in constraints:
            if isinstance(constraint, FixedWeightConstraint):
                rows.append(
                    self._fixed_weight_constraint_row(
                        n_total_sites, constraint, site_to_index
                    )
                )
            elif isinstance(constraint, RatioConstraint):
                rows.append(
                    self._ratio_constraint_row(n_total_sites, constraint, site_to_index)
                )
        return rows
