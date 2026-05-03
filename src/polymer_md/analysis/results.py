from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
    ImproperParameter,
)

AnalysisResult = dict[Fragment, dict[ForceFieldParameter, list[float]]]

_PARAMETER_CLASSES: tuple[type, ...] = (
    BondParameter,
    AngleParameter,
    DihedralParameter,
    ImproperParameter,
    AtomParameter,
)

_SCHEMA_VERSION = "1"


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

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "results": _serialise_results(self.results),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> ComparisonResult:
        payload = json.loads(Path(path).read_text())
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"Cannot load ComparisonResult: unknown schema version {version!r}. "
                f"Expected {_SCHEMA_VERSION!r}."
            )
        return cls(results=_deserialise_results(payload["results"]))


def _serialise_results(results: dict[str, AnalysisResult]) -> dict:
    return {
        label: _serialise_analysis(analysis)
        for label, analysis in results.items()
    }


def _serialise_analysis(analysis: AnalysisResult) -> dict:
    return {
        fragment.pattern: _serialise_param_values(param_values)
        for fragment, param_values in analysis.items()
    }


def _serialise_param_values(param_values: dict[ForceFieldParameter, list[float]]) -> dict:
    return {
        _parameter_key(parameter): values
        for parameter, values in param_values.items()
    }


def _parameter_key(parameter: ForceFieldParameter) -> str:
    return f"{type(parameter).__name__}.{parameter.name}"


def _deserialise_results(raw: dict) -> dict[str, AnalysisResult]:
    return {
        label: _deserialise_analysis(analysis)
        for label, analysis in raw.items()
    }


def _deserialise_analysis(raw: dict) -> AnalysisResult:
    return {
        Fragment(pattern=pattern): _deserialise_param_values(param_values)
        for pattern, param_values in raw.items()
    }


def _deserialise_param_values(raw: dict) -> dict[ForceFieldParameter, list[float]]:
    return {
        _parameter_from_key(key): values
        for key, values in raw.items()
    }


def _parameter_from_key(key: str) -> ForceFieldParameter:
    class_name, member_name = key.split(".", 1)
    for parameter_class in _PARAMETER_CLASSES:
        if parameter_class.__name__ == class_name:
            return parameter_class[member_name]
    raise ValueError(
        f"Unknown parameter key {key!r}. "
        f"Expected one of: {[c.__name__ for c in _PARAMETER_CLASSES]}."
    )
