from __future__ import annotations


class MeanStrategy:
    def resolve(self, values: list[float]) -> float:
        return sum(values) / len(values)


class FirstStrategy:
    def resolve(self, values: list[float]) -> float:
        return values[0]


class MaxStrategy:
    def resolve(self, values: list[float]) -> float:
        return max(values)
