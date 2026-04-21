from __future__ import annotations

from polymer_md.parameterisation.fragments.data_models.match import ResolutionStrategy


class MeanStrategy(ResolutionStrategy):
    def resolve(self, values: list[float]) -> float:
        return sum(values) / len(values)


class FirstStrategy(ResolutionStrategy):
    def resolve(self, values: list[float]) -> float:
        return values[0]


class MaxStrategy(ResolutionStrategy):
    def resolve(self, values: list[float]) -> float:
        return max(values)
