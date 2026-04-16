from __future__ import annotations

from dataclasses import dataclass

from polymer_md.building.data_models.residue import AdditionPolymerResidue


@dataclass(frozen=True)
class MonomerSpec:
    residue: AdditionPolymerResidue
    weight: float

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")

    @property
    def residue_id(self) -> str:
        return self.residue.id
