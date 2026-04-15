from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SiteKey:
    residue_id: str
    site_index: int

    def __str__(self) -> str:
        return f"{self.residue_id}:{self.site_index}"


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
        T = self._normalised()
        i = self.site_to_idx[from_site]
        return {sk: float(T[i, j]) for j, sk in enumerate(self.site_keys)}

    def sample_next(self, from_site: SiteKey, rng: np.random.Generator) -> SiteKey:
        probs = self.get_probabilities(from_site)
        keys = list(probs.keys())
        p = np.array([probs[k] for k in keys])
        return keys[rng.choice(len(keys), p=p)]

    def stationary_distribution(self) -> dict[SiteKey, float]:
        T = self._normalised()
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()
        return {sk: float(pi[i]) for i, sk in enumerate(self.site_keys)}
