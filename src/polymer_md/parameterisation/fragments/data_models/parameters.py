from __future__ import annotations

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
    PHASE = auto()
    PERIODICITY = auto()


class AtomParameter(Enum):
    CHARGE = auto()
    EPSILON = auto()
    SIGMA = auto()
    MASS = auto()


ForceFieldParameter = Union[BondParameter, AngleParameter, DihedralParameter, AtomParameter]
