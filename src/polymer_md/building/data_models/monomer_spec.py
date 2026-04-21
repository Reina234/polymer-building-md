from __future__ import annotations

from dataclasses import dataclass

from polymer_md.building.data_models.residue import AdditionPolymerResidue


@dataclass(frozen=True)
class MonomerSpec:
    residue: AdditionPolymerResidue
    weight: float
    ht_fraction: float = 0.95

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")
        if not 0.0 <= self.ht_fraction <= 1.0:
            raise ValueError(
                f"ht_fraction must be between 0 and 1, got {self.ht_fraction}"
            )

    @property
    def residue_id(self) -> str:
        return self.residue.id
