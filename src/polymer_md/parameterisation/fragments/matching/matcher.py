from __future__ import annotations

import parmed as pmd
from rdkit import Chem

from polymer_md.utils.parmed_helper import ParmedTypeResolver
from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
    AnnotatedMember,
)
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import (
    FragmentMatch,
    ParameterHit,
    ParameterRecord,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
)


class FragmentMatcher:
    def __init__(self, fragments: list[Fragment]) -> None:
        self._fragments = fragments

    def match_all(self, mol: Chem.Mol) -> list[FragmentMatch]:
        return [
            match
            for fragment in self._fragments
            for match in self._match_fragment(mol, fragment)
        ]

    def build_records(
        self,
        matches: list[FragmentMatch],
        structure: pmd.Structure,
    ) -> list[ParameterRecord]:
        hits_by_key: dict[tuple, list[ParameterHit]] = {}
        for match_instance, match in enumerate(matches):
            self._collect_hits(match, match_instance, structure, hits_by_key)
        return self._assemble_records(hits_by_key)

    @staticmethod
    def _match_fragment(mol: Chem.Mol, fragment: Fragment) -> list[FragmentMatch]:
        query = Chem.MolFromSmarts(fragment.pattern)
        return [
            FragmentMatch(
                fragment=fragment,
                atom_map=tuple(enumerate(rdkit_match)),
            )
            for rdkit_match in mol.GetSubstructMatches(query)
        ]

    @staticmethod
    def _collect_hits(
        match: FragmentMatch,
        match_instance: int,
        structure: pmd.Structure,
        hits_by_key: dict[tuple, list[ParameterHit]],
    ) -> None:
        for member in match.fragment.all_members:
            global_indices = match.global_indices_for(member)
            value = FragmentMatcher._extract_value(structure, global_indices, member)
            key = (global_indices, member.parameter)
            hit = ParameterHit(
                value=value,
                fragment=match.fragment,
                match_instance=match_instance,
                member_local_indices=FragmentMatcher._local_indices_of(member),
            )
            hits_by_key.setdefault(key, []).append(hit)

    @staticmethod
    def _assemble_records(
        hits_by_key: dict[tuple, list[ParameterHit]],
    ) -> list[ParameterRecord]:
        return [
            ParameterRecord(
                global_indices=global_indices,
                parameter=parameter,
                hits=tuple(hits),
            )
            for (global_indices, parameter), hits in hits_by_key.items()
        ]

    @staticmethod
    def _local_indices_of(member: AnnotatedMember) -> tuple[int, ...]:
        if isinstance(member, AnnotatedAtom):
            return (member.local_index,)
        return member.local_indices

    @staticmethod
    def _extract_value(
        structure: pmd.Structure,
        global_indices: tuple[int, ...],
        member: AnnotatedMember,
    ) -> float:
        if isinstance(member, AnnotatedBond):
            return FragmentMatcher._bond_value(structure, global_indices, member.parameter)
        if isinstance(member, AnnotatedAngle):
            return FragmentMatcher._angle_value(structure, global_indices, member.parameter)
        if isinstance(member, AnnotatedDihedral):
            return FragmentMatcher._dihedral_value(structure, global_indices, member.parameter)
        return FragmentMatcher._atom_value(structure, global_indices[0], member.parameter)

    @staticmethod
    def _bond_value(
        structure: pmd.Structure,
        atom_indices: tuple[int, ...],
        parameter: BondParameter,
    ) -> float:
        index_set = frozenset(atom_indices)
        for bond in structure.bonds:
            if frozenset({bond.atom1.idx, bond.atom2.idx}) == index_set:
                return FragmentMatcher._read_bond_parameter(bond, parameter)
        raise ValueError(f"Bond not found between atoms {atom_indices}")

    @staticmethod
    def _read_bond_parameter(bond: pmd.Bond, parameter: BondParameter) -> float:
        bond_type = ParmedTypeResolver.bond_type(bond)
        if parameter == BondParameter.FORCE_CONSTANT:
            return float(bond_type.k)
        if parameter == BondParameter.EQUILIBRIUM_LENGTH:
            return float(bond_type.req)
        raise ValueError(f"Unknown bond parameter: {parameter}")  # pragma: no cover

    @staticmethod
    def _angle_value(
        structure: pmd.Structure,
        atom_indices: tuple[int, ...],
        parameter: AngleParameter,
    ) -> float:
        index_set = frozenset(atom_indices)
        for angle in structure.angles:
            if frozenset({angle.atom1.idx, angle.atom2.idx, angle.atom3.idx}) == index_set:
                return FragmentMatcher._read_angle_parameter(angle, parameter)
        raise ValueError(f"Angle not found for atoms {atom_indices}")

    @staticmethod
    def _read_angle_parameter(angle: pmd.Angle, parameter: AngleParameter) -> float:
        angle_type = ParmedTypeResolver.angle_type(angle)
        if parameter == AngleParameter.FORCE_CONSTANT:
            return float(angle_type.k)
        if parameter == AngleParameter.EQUILIBRIUM_ANGLE:
            return float(angle_type.theteq)
        raise ValueError(f"Unknown angle parameter: {parameter}")  # pragma: no cover

    @staticmethod
    def _dihedral_value(
        structure: pmd.Structure,
        atom_indices: tuple[int, ...],
        parameter: DihedralParameter,
    ) -> float:
        index_set = frozenset(atom_indices)
        for dihedral in structure.dihedrals:
            indices = {dihedral.atom1.idx, dihedral.atom2.idx,
                       dihedral.atom3.idx, dihedral.atom4.idx}
            if frozenset(indices) == index_set:
                return FragmentMatcher._read_dihedral_parameter(dihedral, parameter)
        raise ValueError(f"Dihedral not found for atoms {atom_indices}")

    @staticmethod
    def _read_dihedral_parameter(dihedral: pmd.Dihedral, parameter: DihedralParameter) -> float:
        dihedral_type = ParmedTypeResolver.dihedral_type(dihedral)
        if parameter == DihedralParameter.FORCE_CONSTANT:
            return float(dihedral_type.phi_k)
        if parameter == DihedralParameter.PHASE:
            return float(dihedral_type.phase)
        if parameter == DihedralParameter.PERIODICITY:
            return float(dihedral_type.per)
        raise ValueError(f"Unknown dihedral parameter: {parameter}")  # pragma: no cover

    @staticmethod
    def _atom_value(
        structure: pmd.Structure,
        atom_index: int,
        parameter: AtomParameter,
    ) -> float:
        atom = structure.atoms[atom_index]
        if parameter == AtomParameter.CHARGE:
            return float(atom.charge)
        if parameter == AtomParameter.EPSILON:
            return float(atom.epsilon)
        if parameter == AtomParameter.SIGMA:
            return float(atom.sigma)
        if parameter == AtomParameter.MASS:
            return float(atom.mass)
        raise ValueError(f"Unknown atom parameter: {parameter}")  # pragma: no cover
