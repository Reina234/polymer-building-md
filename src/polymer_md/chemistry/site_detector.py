from __future__ import annotations

import logging
from functools import lru_cache

from rdkit import Chem

from polymer_parameterisation.core.monomer import Monomer
from polymer_parameterisation.core.sites import DetectedSite
from polymer_parameterisation.chemistry.registry import ChemistryRegistry

logger = logging.getLogger(__name__)


class SiteDetector:
    def __init__(self, registry: ChemistryRegistry) -> None:
        registry.validate()
        self._registry = registry
        self._compiled_patterns = self._compile_patterns()

    def detect(self, monomer: Monomer) -> list[DetectedSite]:
        mol = Chem.MolFromSmiles(monomer.smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        return self._detect_cached(canonical_smiles)

    def detect_in_construct_mol(
        self,
        mol: Chem.Mol,
        restricted_atom_indices: frozenset[int] | set[int],
    ) -> list[DetectedSite]:
        raw_sites = self._detect_all_sites_in_mol(mol)
        return [
            site for site in raw_sites
            if set(site.atom_indices).issubset(restricted_atom_indices)
        ]

    @lru_cache(maxsize=256)
    def _detect_cached(self, canonical_smiles: str) -> list[DetectedSite]:
        mol = Chem.MolFromSmiles(canonical_smiles)
        raw_sites = self._detect_all_sites_in_mol(mol)
        return self._resolve_overlapping_sites(raw_sites)

    def _detect_all_sites_in_mol(self, mol: Chem.Mol) -> list[DetectedSite]:
        all_sites: list[DetectedSite] = []
        for group in self._registry.groups:
            pattern = self._compiled_patterns[group.name]
            matches = mol.GetSubstructMatches(pattern)
            for match_index, atom_indices in enumerate(matches):
                site = DetectedSite(
                    group_name=group.name,
                    match_index=match_index,
                    atom_indices=tuple(atom_indices),
                )
                all_sites.append(site)
        return all_sites

    def _resolve_overlapping_sites(self, sites: list[DetectedSite]) -> list[DetectedSite]:
        sites_to_drop: set[int] = set()
        for i, site_a in enumerate(sites):
            for j, site_b in enumerate(sites):
                if i >= j:
                    continue
                self._check_pair_for_overlap(i, j, site_a, site_b, sites_to_drop)
        return [site for index, site in enumerate(sites) if index not in sites_to_drop]

    def _check_pair_for_overlap(
        self,
        index_a: int,
        index_b: int,
        site_a: DetectedSite,
        site_b: DetectedSite,
        sites_to_drop: set[int],
    ) -> None:
        atoms_a = frozenset(site_a.atom_indices)
        atoms_b = frozenset(site_b.atom_indices)
        intersection = atoms_a & atoms_b

        if not intersection:
            return

        if atoms_a == atoms_b:
            logger.warning(
                "Groups '%s' and '%s' matched identical atom sets. "
                "Consider consolidating into one GroupType with multiple ReactionOutcomes.",
                site_a.group_name,
                site_b.group_name,
            )
            return

        a_contains_b = atoms_b.issubset(atoms_a)
        b_contains_a = atoms_a.issubset(atoms_b)
        group_a = self._registry._groups[site_a.group_name]
        group_b = self._registry._groups[site_b.group_name]

        if a_contains_b and site_b.group_name in group_a.overrides:
            sites_to_drop.add(index_b)
            return

        if b_contains_a and site_a.group_name in group_b.overrides:
            sites_to_drop.add(index_a)
            return

        if a_contains_b or b_contains_a:
            logger.warning(
                "Group '%s' contains group '%s' but no override is declared. "
                "Add an override or refine your SMARTS to avoid ambiguity.",
                site_a.group_name if a_contains_b else site_b.group_name,
                site_b.group_name if a_contains_b else site_a.group_name,
            )
            return

        logger.info(
            "Partial overlap between groups '%s' and '%s'. "
            "Both are kept as competing open sites.",
            site_a.group_name,
            site_b.group_name,
        )

    def _compile_patterns(self) -> dict[str, Chem.Mol]:
        patterns: dict[str, Chem.Mol] = {}
        for group in self._registry.groups:
            pattern = Chem.MolFromSmarts(group.smarts)
            if pattern is None:
                raise ValueError(
                    f"Invalid SMARTS for group '{group.name}': {group.smarts!r}"
                )
            patterns[group.name] = pattern
        return patterns
