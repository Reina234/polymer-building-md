from enum import IntEnum
from typing import Literal


class MapLabels(IntEnum):
    HEAD = 1
    TAIL = 2
    CAP = 3

    def other(self) -> "MapLabels":
        if self == MapLabels.HEAD:
            return MapLabels.TAIL
        if self == MapLabels.TAIL:
            return MapLabels.HEAD
        raise ValueError(f"No other site defined for {self}")


PolymerisationLabels = Literal[MapLabels.HEAD, MapLabels.TAIL]
