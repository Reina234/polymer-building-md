from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonomerComposition:
    residue_id: str
    weight: float

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")
