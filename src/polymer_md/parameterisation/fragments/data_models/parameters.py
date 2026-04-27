from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


class BondParameter(Enum):
    FORCE_CONSTANT = auto()
    EQUILIBRIUM_LENGTH = auto()


class AngleParameter(Enum):
    FORCE_CONSTANT = auto()
    EQUILIBRIUM_ANGLE = auto()


class DihedralParameter(Enum):
    FORCE_CONSTANT = auto()


class AtomParameter(Enum):
    CHARGE = auto()
    EPSILON = auto()
    SIGMA = auto()
    MASS = auto()


class ImproperParameter(Enum):
    FORCE_CONSTANT = auto()


ForceFieldParameter = Union[BondParameter, AngleParameter, DihedralParameter, ImproperParameter, AtomParameter]


@dataclass(frozen=True)
class DihedralTerm:
    force_constant: float
    phase: float
    periodicity: float
