from __future__ import annotations

from dataclasses import dataclass, field

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedAngle,
    AnnotatedDihedral,
    AnnotatedImproper,
    AnnotatedMember,
)


@dataclass(frozen=True)
class Fragment:
    pattern: str
    annotated_bonds: tuple[AnnotatedBond, ...] = field(default_factory=tuple)
    annotated_angles: tuple[AnnotatedAngle, ...] = field(default_factory=tuple)
    annotated_dihedrals: tuple[AnnotatedDihedral, ...] = field(default_factory=tuple)
    annotated_impropers: tuple[AnnotatedImproper, ...] = field(default_factory=tuple)
    annotated_atoms: tuple[AnnotatedAtom, ...] = field(default_factory=tuple)

    @property
    def all_members(self) -> tuple[AnnotatedMember, ...]:
        return (
            *self.annotated_bonds,
            *self.annotated_angles,
            *self.annotated_dihedrals,
            *self.annotated_impropers,
            *self.annotated_atoms,
        )
