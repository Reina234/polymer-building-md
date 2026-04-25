from __future__ import annotations

import logging
from dataclasses import dataclass, field

import parmed as pmd
from rdkit import Chem

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
    AnnotatedMember,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
)
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.fragments.matching.resolution import MeanStrategy
from polymer_md.parameterisation.fragments.data_models.match import ResolutionStrategy
from polymer_md.parameterisation.strategies.base import MissingParameterError, MissingParameterStrategy, StrategyContext
from polymer_md.parameterisation.strategies.strict import StrictMissingParameterStrategy

logger = logging.getLogger(__name__)


@dataclass
class PolymerParameterisationTiler:
    library: FragmentLibrary
    resolution_strategy: ResolutionStrategy = field(default_factory=MeanStrategy)
    missing_strategies: dict[type, MissingParameterStrategy] = field(default_factory=dict)

    def tile(
        self,
        structure: pmd.Structure,
        derived_mol: Chem.Mol,
        polymer_atom_metadata: dict[int, tuple[str, int]] | None = None,
    ) -> pmd.Structure:
        context = StrategyContext(
            library=self.library,
            derived_mol=derived_mol,
            polymer_atom_metadata=polymer_atom_metadata or {},
        )
        assignments = self._collect_assignments(derived_mol)
        gaff2_types = self._collect_gaff2_types(derived_mol)
        self._apply_atoms(structure, assignments, gaff2_types, context)
        self._apply_bonds(structure, assignments, context)
        self._apply_angles(structure, assignments, context)
        self._apply_dihedrals(structure, assignments, context)
        return structure

    def _collect_assignments(self, derived_mol: Chem.Mol) -> dict[tuple, list[float]]:
        assignments: dict[tuple, list[float]] = {}
        for fragment in self.library.to_fragments():
            query = Chem.MolFromSmarts(fragment.pattern)
            if query is None:
                continue
            for rdkit_match in derived_mol.GetSubstructMatches(query):
                for member in fragment.all_members:
                    global_indices = self._member_global_indices(member, rdkit_match)
                    local_indices = self._member_local_indices(member)
                    values = self.library.values_for_member(
                        fragment.pattern, local_indices, member.parameter
                    )
                    key = (self._canonical_key(global_indices, member), member.parameter)
                    assignments.setdefault(key, []).extend(values)
        return assignments

    def _collect_gaff2_types(self, derived_mol: Chem.Mol) -> dict[int, str]:
        gaff2_types: dict[int, str] = {}
        for pattern, local_map in self.library.atom_metadata.items():
            query = Chem.MolFromSmarts(pattern)
            if query is None:
                continue
            for rdkit_match in derived_mol.GetSubstructMatches(query):
                for local_idx, meta in local_map.items():
                    if local_idx < len(rdkit_match) and meta.gaff2_type:
                        global_idx = rdkit_match[local_idx]
                        gaff2_types.setdefault(global_idx, meta.gaff2_type)
        return gaff2_types

    def _apply_atoms(
        self,
        structure: pmd.Structure,
        assignments: dict,
        gaff2_types: dict[int, str],
        context: StrategyContext,
    ) -> None:
        for atom in structure.atoms:
            for parameter in AtomParameter:
                key = ((atom.idx,), parameter)
                values = assignments.get(key, [])
                value = self._resolve_or_missing((atom.idx,), parameter, values, context)
                self._set_atom_param(atom, parameter, value)
            if atom.idx in gaff2_types:
                atom.type = gaff2_types[atom.idx]

    def _apply_bonds(
        self,
        structure: pmd.Structure,
        assignments: dict,
        context: StrategyContext,
    ) -> None:
        for bond in structure.bonds:
            i, j = bond.atom1.idx, bond.atom2.idx
            canonical = (min(i, j), max(i, j))
            k = self._resolve_or_missing(
                canonical, BondParameter.FORCE_CONSTANT,
                assignments.get((canonical, BondParameter.FORCE_CONSTANT), []), context
            )
            req = self._resolve_or_missing(
                canonical, BondParameter.EQUILIBRIUM_LENGTH,
                assignments.get((canonical, BondParameter.EQUILIBRIUM_LENGTH), []), context
            )
            bond.type = pmd.BondType(k=k, req=req)

    def _apply_angles(
        self,
        structure: pmd.Structure,
        assignments: dict,
        context: StrategyContext,
    ) -> None:
        for angle in structure.angles:
            i, j, k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
            canonical = (min(i, k), j, max(i, k))
            force_k = self._resolve_or_missing(
                canonical, AngleParameter.FORCE_CONSTANT,
                assignments.get((canonical, AngleParameter.FORCE_CONSTANT), []), context
            )
            theteq = self._resolve_or_missing(
                canonical, AngleParameter.EQUILIBRIUM_ANGLE,
                assignments.get((canonical, AngleParameter.EQUILIBRIUM_ANGLE), []), context
            )
            angle.type = pmd.AngleType(k=force_k, theteq=theteq)

    def _apply_dihedrals(
        self,
        structure: pmd.Structure,
        assignments: dict,
        context: StrategyContext,
    ) -> None:
        for dihedral in structure.dihedrals:
            if dihedral.improper:
                continue
            i, j, k, l = (
                dihedral.atom1.idx, dihedral.atom2.idx,
                dihedral.atom3.idx, dihedral.atom4.idx,
            )
            forward = (i, j, k, l)
            reverse = (l, k, j, i)
            canonical = min(forward, reverse)
            phi_k = self._resolve_or_missing(
                canonical, DihedralParameter.FORCE_CONSTANT,
                assignments.get((canonical, DihedralParameter.FORCE_CONSTANT), []), context
            )
            phase = self._resolve_or_missing(
                canonical, DihedralParameter.PHASE,
                assignments.get((canonical, DihedralParameter.PHASE), []), context
            )
            per = self._resolve_or_missing(
                canonical, DihedralParameter.PERIODICITY,
                assignments.get((canonical, DihedralParameter.PERIODICITY), []), context
            )
            dihedral.type = pmd.DihedralType(phi_k=phi_k, phase=phase, per=per)

    def _resolve_or_missing(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        values: list[float],
        context: StrategyContext,
    ) -> float:
        if values:
            return self.resolution_strategy.resolve(values)
        strategy = self.missing_strategies.get(type(parameter), StrictMissingParameterStrategy())
        return strategy.resolve(global_indices, parameter, context)

    @staticmethod
    def _canonical_key(
        global_indices: tuple[int, ...],
        member: AnnotatedMember,
    ) -> tuple[int, ...]:
        if isinstance(member, AnnotatedAtom):
            return global_indices
        if isinstance(member, AnnotatedBond):
            return (min(global_indices), max(global_indices))
        if isinstance(member, AnnotatedAngle):
            i, j, k = global_indices
            return (min(i, k), j, max(i, k))
        forward = global_indices
        reverse = global_indices[::-1]
        return min(forward, reverse)

    @staticmethod
    def _member_global_indices(
        member: AnnotatedMember,
        rdkit_match: tuple[int, ...],
    ) -> tuple[int, ...]:
        if isinstance(member, AnnotatedAtom):
            return (rdkit_match[member.local_index],)
        return tuple(rdkit_match[i] for i in member.local_indices)

    @staticmethod
    def _member_local_indices(member: AnnotatedMember) -> tuple[int, ...]:
        if isinstance(member, AnnotatedAtom):
            return (member.local_index,)
        return member.local_indices

    @staticmethod
    def _set_atom_param(atom: pmd.Atom, parameter: AtomParameter, value: float) -> None:
        if parameter == AtomParameter.CHARGE:
            atom.charge = value
        elif parameter == AtomParameter.EPSILON:
            atom.epsilon = value
        elif parameter == AtomParameter.SIGMA:
            atom.sigma = value
        elif parameter == AtomParameter.MASS:
            atom.mass = value


def adjust_charge_neutrality(structure: pmd.Structure) -> None:
    total = sum(atom.charge for atom in structure.atoms)
    if abs(total) < 1e-10:
        return
    per_atom = total / len(structure.atoms)
    for atom in structure.atoms:
        atom.charge -= per_atom
