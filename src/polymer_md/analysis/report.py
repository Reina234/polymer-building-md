from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polymer_md.analysis.results import ComparisonResult, Summary
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import ForceFieldParameter


_HEADER = f"{'Fragment':<24} {'Parameter':<26} {'Label':<18} {'Mean':>10} {'Std':>9} {'Min':>10} {'Max':>10} {'n':>5}"
_DIVIDER = "-" * len(_HEADER)


@dataclass
class TextReport:
    result: ComparisonResult
    title: str = "Parameter Comparison"

    def __str__(self) -> str:
        lines = self._build_lines()
        return "\n".join(lines)

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(self) + "\n")

    def _build_lines(self) -> list[str]:
        summary = self.result.summarise()
        lines = [self.title, _DIVIDER, _HEADER, _DIVIDER]
        for fragment, parameter in _ordered_fragment_parameter_pairs(summary):
            lines.extend(self._rows_for_pair(summary, fragment, parameter))
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
        return lines

    def _rows_for_pair(
        self,
        summary: dict,
        fragment: Fragment,
        parameter: ForceFieldParameter,
    ) -> list[str]:
        rows = []
        first = True
        for label, analysis in summary.items():
            stat = analysis.get(fragment, {}).get(parameter)
            if stat is None:
                continue
            fragment_col = fragment.pattern if first else ""
            parameter_col = _parameter_display_name(parameter) if first else ""
            first = False
            rows.append(_format_row(fragment_col, parameter_col, label, stat))
        return rows


def _ordered_fragment_parameter_pairs(
    summary: dict[str, dict[Fragment, dict[ForceFieldParameter, Summary]]],
) -> list[tuple[Fragment, ForceFieldParameter]]:
    seen: set[tuple] = set()
    ordered: list[tuple[Fragment, ForceFieldParameter]] = []
    for analysis in summary.values():
        for fragment, param_map in analysis.items():
            for parameter in param_map:
                key = (fragment.pattern, type(parameter).__name__, parameter)
                if key not in seen:
                    seen.add(key)
                    ordered.append((fragment, parameter))
    return ordered


def _parameter_display_name(parameter: ForceFieldParameter) -> str:
    return f"{type(parameter).__name__}.{parameter.name}"


def _format_row(
    fragment: str,
    parameter: str,
    label: str,
    stat: Summary,
) -> str:
    return (
        f"{fragment:<24} {parameter:<26} {label:<18}"
        f" {stat.mean:>10.4f} {stat.std:>9.4f}"
        f" {stat.min:>10.4f} {stat.max:>10.4f} {stat.count:>5}"
    )
