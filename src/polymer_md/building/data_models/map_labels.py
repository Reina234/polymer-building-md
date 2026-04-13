from enum import IntEnum
from typing import Literal


class MapLabels(IntEnum):
    HEAD = 1
    TAIL = 2
    INACTIVE_END = 3
    CAP_GROWING = 4
    CAP_INACTIVE = 5
    INCOMING = 6

    def other(self) -> "MapLabels":
        if self == MapLabels.HEAD:
            return MapLabels.TAIL
        if self == MapLabels.TAIL:
            return MapLabels.HEAD
        raise ValueError(f"No other site defined for {self}")


PolymerisationLabels = Literal[MapLabels.HEAD, MapLabels.TAIL]
