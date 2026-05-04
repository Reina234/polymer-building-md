from __future__ import annotations

import numpy as np

from polymer_md.building.data_models.map_labels import (
    SITE_INDEX_TO_POLYMERISATION_LABEL,
    PolymerisationLabels,
)
from polymer_md.building.data_models.residue import AdditionPolymerResidue
from polymer_md.building.data_models.transition_matrix import SiteKey, TransitionMatrix
from polymer_md.building.growing_polymer import AdditionPolymer
from polymer_md.core.caps import Cap
from polymer_md.core.polymer import Polymer


class RandomPolymerBuilder:
    def __init__(
        self,
        residues: dict[str, AdditionPolymerResidue],
        matrix: TransitionMatrix,
        cap: Cap,
    ) -> None:
        self._residues = residues
        self._matrix = matrix
        self._cap = cap
        self._validate_residues_covered()
        self._validate_site_indices()

    def build(self, n: int, rng: np.random.Generator) -> Polymer:
        polymer, _ = self.build_with_connections(n, rng)
        return polymer

    def build_with_connections(
        self, n: int, rng: np.random.Generator
    ) -> tuple[Polymer, list[tuple[int, int]]]:
        """Returns (polymer, connections) where connections[i] = (from_site_index, to_site_index)
        is the physical bond between monomer i and monomer i+1.

        from_site_index is the growing-end site of monomer i (the atom that forms the bond).
        to_site_index is the incoming site of monomer i+1 (the atom that receives the bond).
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        ap = AdditionPolymer()
        initial_site = self._sample_initial_site(rng)
        self._initialise(ap, initial_site)
        active_site = self._complement_site(initial_site)
        # Physical growing-end site of the current chain end.
        # For the initial monomer this is initial_site (not active_site, which is its complement).
        # For every subsequently added monomer growing_end == active_site.
        growing_end_site_index = initial_site.site_index
        connections: list[tuple[int, int]] = []
        for _ in range(n - 1):
            next_site = self._matrix.sample_next(active_site, rng)
            connections.append((growing_end_site_index, next_site.site_index))
            ap.add(
                self._residues[next_site.residue_id],
                site=self._to_map_label(next_site),
            )
            active_site = self._complement_site(next_site)
            growing_end_site_index = active_site.site_index
        return ap.export(self._cap), connections

    def _initialise(self, polymer: AdditionPolymer, site: SiteKey) -> None:
        polymer.initialise(
            self._residues[site.residue_id],
            site=self._to_map_label(site),
        )

    def _grow_one(
        self,
        polymer: AdditionPolymer,
        active_site: SiteKey,
        rng: np.random.Generator,
    ) -> SiteKey:
        next_site = self._matrix.sample_next(active_site, rng)
        polymer.add(
            self._residues[next_site.residue_id],
            site=self._to_map_label(next_site),
        )
        return self._complement_site(next_site)

    def _sample_initial_site(self, rng: np.random.Generator) -> SiteKey:
        stationary = self._matrix.stationary_distribution()
        site_keys = list(stationary.keys())
        probabilities = np.array([stationary[sk] for sk in site_keys])
        chosen_index = rng.choice(len(site_keys), p=probabilities)
        return site_keys[chosen_index]

    @staticmethod
    def _complement_site(site: SiteKey) -> SiteKey:
        return SiteKey(site.residue_id, 1 - site.site_index)

    @staticmethod
    def _to_map_label(site: SiteKey) -> PolymerisationLabels:
        return SITE_INDEX_TO_POLYMERISATION_LABEL[site.site_index]

    def _validate_residues_covered(self) -> None:
        matrix_residue_ids = {sk.residue_id for sk in self._matrix.site_keys}
        missing = matrix_residue_ids - self._residues.keys()
        if missing:
            raise ValueError(
                f"Residue IDs referenced in matrix but not provided: {missing}"
            )

    def _validate_site_indices(self) -> None:
        for site_key in self._matrix.site_keys:
            if site_key.site_index not in SITE_INDEX_TO_POLYMERISATION_LABEL:
                raise ValueError(
                    f"site_index {site_key.site_index} on '{site_key.residue_id}' "
                    f"is not a valid addition polymer site (expected 0 or 1)."
                )
