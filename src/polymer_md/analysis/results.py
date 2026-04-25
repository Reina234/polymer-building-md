from __future__ import annotations

import statistics
from dataclasses import dataclass

from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import ForceFieldParameter

AnalysisResult = dict[Fragment, dict[ForceFieldParameter, list[float]]]


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    count: int
    min: float
    max: float


@dataclass
class ComparisonResult:
    results: dict[str, AnalysisResult]

    def summarise(self) -> dict[str, dict[Fragment, dict[ForceFieldParameter, Summary]]]:
        summary: dict[str, dict[Fragment, dict[ForceFieldParameter, Summary]]] = {}
        for label, analysis in self.results.items():
            summary[label] = {}
            for fragment, param_values in analysis.items():
                summary[label][fragment] = {}
                for param, values in param_values.items():
                    if not values:
                        continue
                    summary[label][fragment][param] = Summary(
                        mean=statistics.mean(values),
                        std=statistics.stdev(values) if len(values) > 1 else 0.0,
                        count=len(values),
                        min=min(values),
                        max=max(values),
                    )
        return summary
