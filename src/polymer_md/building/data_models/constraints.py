from __future__ import annotations

from dataclasses import dataclass

from polymer_md.building.data_models.transition_matrix import SiteKey


@dataclass(frozen=True)
class RatioConstraint:
    from_a: SiteKey
    to_a: SiteKey
    from_b: SiteKey
    to_b: SiteKey
    ratio: float

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError(f"ratio must be positive, got {self.ratio}")


@dataclass(frozen=True)
class FixedWeightConstraint:
    from_site: SiteKey
    to_site: SiteKey
    weight: float

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


TransitionConstraint = RatioConstraint | FixedWeightConstraint
