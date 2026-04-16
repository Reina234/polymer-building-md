from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SiteKey:
    residue_id: str
    site_index: int

    def __str__(self) -> str:
        return f"{self.residue_id}:{self.site_index}"

    @classmethod
    def head(cls, residue_id: str) -> SiteKey:
        return cls(residue_id, 0)

    @classmethod
    def tail(cls, residue_id: str) -> SiteKey:
        return cls(residue_id, 1)


@dataclass(frozen=True)
class TransitionMatrix:
    site_keys: tuple[SiteKey, ...]
    weights: np.ndarray  # shape (N, N), weights[i, j] >= 0
    site_to_idx: dict[SiteKey, int] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        n = len(self.site_keys)
        if self.weights.shape != (n, n):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match "
                f"number of sites ({n}x{n})"
            )
        if np.any(self.weights < 0):
            raise ValueError("All weights must be non-negative.")
        object.__setattr__(
            self, "site_to_idx", {sk: i for i, sk in enumerate(self.site_keys)}
        )

    @property
    def n_sites(self) -> int:
        return len(self.site_keys)

    def _normalised(self) -> np.ndarray:
        row_sums = self.weights.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, self.weights / row_sums, 1.0 / self.n_sites)

    def get_probabilities(self, from_site: SiteKey) -> dict[SiteKey, float]:
        normalised = self._normalised()
        i = self.site_to_idx[from_site]
        return {sk: float(normalised[i, j]) for j, sk in enumerate(self.site_keys)}

    def sample_next(self, from_site: SiteKey, rng: np.random.Generator) -> SiteKey:
        probabilities = self.get_probabilities(from_site)
        site_keys = list(probabilities.keys())
        weights = np.array([probabilities[sk] for sk in site_keys])
        return site_keys[rng.choice(len(site_keys), p=weights)]

    def stationary_distribution(self) -> dict[SiteKey, float]:
        normalised = self._normalised()
        eigenvalues, eigenvectors = np.linalg.eig(normalised.T)
        stationary_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        stationary = np.real(eigenvectors[:, stationary_index])
        stationary = np.abs(stationary)
        stationary /= stationary.sum()
        return {sk: float(stationary[i]) for i, sk in enumerate(self.site_keys)}
