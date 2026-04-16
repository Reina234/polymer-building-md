from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
)


@dataclass(frozen=True)
class AnnotatedBond:
    local_indices: tuple[int, int]
    parameter: BondParameter


@dataclass(frozen=True)
class AnnotatedAngle:
    local_indices: tuple[int, int, int]
    parameter: AngleParameter


@dataclass(frozen=True)
class AnnotatedDihedral:
    local_indices: tuple[int, int, int, int]
    parameter: DihedralParameter


@dataclass(frozen=True)
class AnnotatedAtom:
    local_index: int
    parameter: AtomParameter


AnnotatedMember = Union[AnnotatedBond, AnnotatedAngle, AnnotatedDihedral, AnnotatedAtom]
