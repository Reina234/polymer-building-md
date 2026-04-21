from __future__ import annotations

from enum import IntEnum

import numpy as np
from rdkit import Chem

from polymer_md.building.data_models.map_labels import (
    SITE_INDEX_TO_POLYMERISATION_LABEL,
    MapLabels,
)
from polymer_md.building.data_models.residue import AdditionPolymerResidue
from polymer_md.building.data_models.transition_matrix import SiteKey, TransitionMatrix
from polymer_md.building.data_models.trimer import (
    Orientation,
    TRIMER_REGION_TAG,
    TrimerRegion,
    TrimerResult,
)
from polymer_md.core.caps import Cap
from polymer_md.utils.rdkit_helper import RDKitHelper


class _TrimerMapNum(IntEnum):
    LEFT_BOND_SRC = 10
    LEFT_BOND_DST = 11
    RIGHT_BOND_SRC = 12
    RIGHT_BOND_DST = 13
    OPEN_LEFT = 20
    OPEN_RIGHT = 21
    CAP_LEFT = 22
    CAP_RIGHT = 23


_MAP_LABEL_VALUES = {int(m) for m in MapLabels}
assert not any(
    int(t) in _MAP_LABEL_VALUES for t in _TrimerMapNum
), "A _TrimerMapNum value clashes with a MapLabels value."


def _tag_atoms_with_region(mol: Chem.Mol, region: TrimerRegion) -> Chem.Mol:
    rw = Chem.rdchem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0:
            atom.SetIntProp(TRIMER_REGION_TAG, int(region))
    return rw.GetMol()


def _collect_region_indices(mol: Chem.Mol, region: TrimerRegion) -> frozenset[int]:
    return frozenset(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp(TRIMER_REGION_TAG)
        and atom.GetIntProp(TRIMER_REGION_TAG) == int(region)
    )


class _TrimerMolAssembler:
    def __init__(self, cap: Cap) -> None:
        self._cap = cap

    def build_uncapped(
        self,
        left: AdditionPolymerResidue,
        left_site: int,
        central: AdditionPolymerResidue,
        k_site_left: int,
        right: AdditionPolymerResidue,
        right_site: int,
    ) -> Chem.Mol:
        return RDKitHelper.join_many_at_map_nums(
            mol=self._prepare_left(left, left_site),
            mols_to_combine=[
                self._prepare_central(central, k_site_left),
                self._prepare_right(right, right_site),
            ],
            pairs=[
                (_TrimerMapNum.LEFT_BOND_SRC, _TrimerMapNum.LEFT_BOND_DST),
                (_TrimerMapNum.RIGHT_BOND_SRC, _TrimerMapNum.RIGHT_BOND_DST),
            ],
        )

    def attach_caps(self, uncapped: Chem.Mol) -> Chem.Mol:
        return RDKitHelper.join_many_at_map_nums(
            mol=uncapped,
            mols_to_combine=[
                self._make_cap(_TrimerMapNum.CAP_LEFT),
                self._make_cap(_TrimerMapNum.CAP_RIGHT),
            ],
            pairs=[
                (_TrimerMapNum.OPEN_LEFT, _TrimerMapNum.CAP_LEFT),
                (_TrimerMapNum.OPEN_RIGHT, _TrimerMapNum.CAP_RIGHT),
            ],
        )

    def _make_cap(self, map_num: _TrimerMapNum) -> Chem.Mol:
        mol = RDKitHelper.relabel_wildcard(
            Chem.rdmolfiles.MolFromSmiles(self._cap.smiles),
            new_map_num=int(map_num),
        )
        return _tag_atoms_with_region(mol, TrimerRegion.CAP)

    @staticmethod
    def _prepare_left(residue: AdditionPolymerResidue, bonding_site: int) -> Chem.Mol:
        bonding_map = SITE_INDEX_TO_POLYMERISATION_LABEL[bonding_site]
        mol = RDKitHelper.replace_map_num(
            Chem.rdchem.Mol(residue.mol), bonding_map, _TrimerMapNum.LEFT_BOND_SRC
        )
        mol = RDKitHelper.replace_map_num(mol, bonding_map.other(), _TrimerMapNum.OPEN_LEFT)
        return _tag_atoms_with_region(mol, TrimerRegion.LEFT)

    @staticmethod
    def _prepare_central(residue: AdditionPolymerResidue, k_site_left: int) -> Chem.Mol:
        left_map = SITE_INDEX_TO_POLYMERISATION_LABEL[k_site_left]
        mol = RDKitHelper.replace_map_num(
            Chem.rdchem.Mol(residue.mol), left_map, _TrimerMapNum.LEFT_BOND_DST
        )
        mol = RDKitHelper.replace_map_num(mol, left_map.other(), _TrimerMapNum.RIGHT_BOND_SRC)
        return _tag_atoms_with_region(mol, TrimerRegion.CENTRAL)

    @staticmethod
    def _prepare_right(residue: AdditionPolymerResidue, bonding_site: int) -> Chem.Mol:
        bonding_map = SITE_INDEX_TO_POLYMERISATION_LABEL[bonding_site]
        mol = RDKitHelper.replace_map_num(
            Chem.rdchem.Mol(residue.mol), bonding_map, _TrimerMapNum.RIGHT_BOND_DST
        )
        mol = RDKitHelper.replace_map_num(mol, bonding_map.other(), _TrimerMapNum.OPEN_RIGHT)
        return _tag_atoms_with_region(mol, TrimerRegion.RIGHT)


class TrimerBuilder:
    def __init__(
        self,
        residues: dict[str, AdditionPolymerResidue],
        matrix: TransitionMatrix,
        cap: Cap,
    ) -> None:
        self._residues = residues
        self._matrix = matrix
        self._assembler = _TrimerMolAssembler(cap)

    def build_all(self) -> list[TrimerResult]:
        stationary = self._matrix.stationary_distribution()
        seen_smiles: set[str] = set()
        return [
            result
            for central_id in self._residues
            for result in self._enumerate_for_central(
                central_id, stationary, seen_smiles
            )
        ]

    def _enumerate_for_central(
        self,
        central_id: str,
        stationary: dict[SiteKey, float],
        seen_smiles: set[str],
    ) -> list[TrimerResult]:
        central_is_regiosymmetric = RDKitHelper.wildcard_ranks_equal(
            self._residues[central_id].mol
        )
        results = []
        for left_id, right_id in self._left_right_pairs(central_is_regiosymmetric):
            for config in self._bond_configs(central_is_regiosymmetric):
                result = self._try_build(
                    left_id, central_id, right_id, config, stationary, seen_smiles
                )
                if result is not None:
                    results.append(result)
        return results

    def _left_right_pairs(
        self, central_is_regiosymmetric: bool
    ) -> list[tuple[str, str]]:
        all_ids = list(self._residues.keys())
        if not central_is_regiosymmetric:
            return [(left, right) for left in all_ids for right in all_ids]
        return [
            (left, right)
            for left in all_ids
            for right in all_ids
            if not self._is_palindrome_duplicate(left, right)
        ]

    def _try_build(
        self,
        left_id: str,
        central_id: str,
        right_id: str,
        config: tuple[int, int, int],
        stationary: dict[SiteKey, float],
        seen_smiles: set[str],
    ) -> TrimerResult | None:
        k_site_left, left_site, right_site = config
        uncapped = self._assembler.build_uncapped(
            self._residues[left_id],
            left_site,
            self._residues[central_id],
            k_site_left,
            self._residues[right_id],
            right_site,
        )
        canonical = RDKitHelper.canonical_smiles_stripped(uncapped)
        if canonical in seen_smiles:
            return None
        seen_smiles.add(canonical)
        capped = self._assembler.attach_caps(uncapped)
        return TrimerResult(
            left_id=left_id,
            central_id=central_id,
            right_id=right_id,
            orientation=(
                Orientation.HEAD_IN if k_site_left == 0 else Orientation.TAIL_IN
            ),
            mol=capped,
            probability=self._compute_probability(
                left_id,
                left_site,
                central_id,
                k_site_left,
                right_id,
                right_site,
                stationary,
            ),
            left_atom_indices=_collect_region_indices(capped, TrimerRegion.LEFT),
            central_atom_indices=_collect_region_indices(capped, TrimerRegion.CENTRAL),
            right_atom_indices=_collect_region_indices(capped, TrimerRegion.RIGHT),
            cap_atom_indices=_collect_region_indices(capped, TrimerRegion.CAP),
        )

    def _compute_probability(
        self,
        left_id: str,
        left_site: int,
        central_id: str,
        k_site_left: int,
        right_id: str,
        right_site: int,
        stationary: dict[SiteKey, float],
    ) -> float:
        normalised = self._normalised_weights()
        left_active = SiteKey(left_id, 1 - left_site)
        k_active = SiteKey(central_id, 1 - k_site_left)
        return (
            stationary.get(left_active, 0.0)
            * self._transition(
                normalised, left_active, SiteKey(central_id, k_site_left)
            )
            * self._transition(normalised, k_active, SiteKey(right_id, right_site))
        )

    def _normalised_weights(self) -> np.ndarray:
        weights = self._matrix.weights
        row_sums = weights.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, weights / row_sums, 1.0 / self._matrix.n_sites)

    def _transition(
        self, normalised: np.ndarray, from_site: SiteKey, to_site: SiteKey
    ) -> float:
        idx = self._matrix.site_to_idx
        from_index = idx.get(from_site)
        to_index = idx.get(to_site)
        if from_index is None or to_index is None:
            return 0.0
        return float(normalised[from_index, to_index])

    def _is_palindrome_duplicate(self, left_id: str, right_id: str) -> bool:
        left_smiles = RDKitHelper.canonical_smiles_stripped(self._residues[left_id].mol)
        right_smiles = RDKitHelper.canonical_smiles_stripped(
            self._residues[right_id].mol
        )
        return left_smiles > right_smiles

    @staticmethod
    def _bond_configs(central_is_regiosymmetric: bool) -> list[tuple[int, int, int]]:
        k_orientations = [0] if central_is_regiosymmetric else [0, 1]
        return [
            (k_site_left, left_site, right_site)
            for k_site_left in k_orientations
            for left_site in range(2)
            for right_site in range(2)
        ]
