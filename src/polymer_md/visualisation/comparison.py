from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from polymer_md.analysis.results import ComparisonResult
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
)
from polymer_md.visualisation._palette import COMPARISON_PALETTE, clean_ax

_MAX_COLS = 3
_EMPTY_FIGURE_SIZE = (6, 3)
_CELL_WIDTH = 5.5
_CELL_HEIGHT = 4.5
_TITLE_FONT_SIZE = 13
_CELL_FONT_SIZE = 9
_BOX_WIDTH = 0.5
_JITTER_RANGE = 0.18
_SCATTER_SIZE = 18


def plot_comparison(result: ComparisonResult) -> plt.Figure:
    pairs = _collect_pairs(result)
    if not pairs:
        return _empty_figure()

    n_cols = min(_MAX_COLS, len(pairs))
    n_rows = math.ceil(len(pairs) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(_CELL_WIDTH * n_cols, _CELL_HEIGHT * n_rows),
        facecolor="white",
        squeeze=False,
    )

    molecule_labels = list(result.results.keys())
    color_map = {label: COMPARISON_PALETTE[i % len(COMPARISON_PALETTE)]
                 for i, label in enumerate(molecule_labels)}

    for cell_idx, (fragment, parameter) in enumerate(pairs):
        row, col = divmod(cell_idx, n_cols)
        ax = axes[row][col]
        _draw_comparison_cell(ax, result, fragment, parameter, molecule_labels, color_map)

    for cell_idx in range(len(pairs), n_rows * n_cols):
        row, col = divmod(cell_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Parameter Comparison", fontsize=_TITLE_FONT_SIZE, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def _empty_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=_EMPTY_FIGURE_SIZE)
    ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def _collect_pairs(result: ComparisonResult) -> list[tuple[Fragment, ForceFieldParameter]]:
    seen: set[tuple] = set()
    ordered: list[tuple[Fragment, ForceFieldParameter]] = []
    for analysis in result.results.values():
        _collect_pairs_from_analysis(analysis, seen, ordered)
    return ordered


def _collect_pairs_from_analysis(
    analysis: dict,
    seen: set[tuple],
    ordered: list[tuple[Fragment, ForceFieldParameter]],
) -> None:
    for fragment, param_map in analysis.items():
        for parameter, values in param_map.items():
            if not values:
                continue
            key = (fragment.pattern, type(parameter).__name__, parameter)
            if key not in seen:
                seen.add(key)
                ordered.append((fragment, parameter))


def _draw_comparison_cell(
    ax: plt.Axes,
    result: ComparisonResult,
    fragment: Fragment,
    parameter: ForceFieldParameter,
    molecule_labels: list[str],
    color_map: dict[str, str],
) -> None:
    clean_ax(ax)

    per_molecule_values, tick_labels, present_colors = _gather_molecule_values(
        result, fragment, parameter, molecule_labels, color_map
    )

    if not per_molecule_values:
        ax.set_axis_off()
        return

    positions = np.arange(1, len(per_molecule_values) + 1)
    bp = ax.boxplot(
        per_molecule_values,
        positions=positions,
        widths=_BOX_WIDTH,
        patch_artist=True,
        medianprops=dict(color="#333333", linewidth=1.8),
        whiskerprops=dict(color="#888888"),
        capprops=dict(color="#888888"),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#AAAAAA",
                        markeredgecolor="#AAAAAA", alpha=0.6),
        notch=False,
    )
    for patch, color in zip(bp["boxes"], present_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, (values, color) in enumerate(zip(per_molecule_values, present_colors), start=1):
        jitter = np.random.default_rng(i).uniform(-_JITTER_RANGE, _JITTER_RANGE, len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color=color,
            s=_SCATTER_SIZE,
            alpha=0.7,
            zorder=5,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=_CELL_FONT_SIZE)
    ax.set_ylabel(_parameter_label(parameter), fontsize=_CELL_FONT_SIZE)
    ax.set_title(
        f"{fragment.pattern}\n{parameter.name.replace('_', ' ').title()}",
        fontsize=_CELL_FONT_SIZE,
        pad=5,
    )
    ax.grid(axis="y", color="#EBEBEB", linewidth=0.8, zorder=0)


def _gather_molecule_values(
    result: ComparisonResult,
    fragment: Fragment,
    parameter: ForceFieldParameter,
    molecule_labels: list[str],
    color_map: dict[str, str],
) -> tuple[list[list[float]], list[str], list[str]]:
    per_molecule_values = []
    tick_labels = []
    present_colors = []
    for label in molecule_labels:
        values = _find_values_for_fragment(result.results.get(label, {}), fragment, parameter)
        if values:
            per_molecule_values.append(values)
            tick_labels.append(label)
            present_colors.append(color_map[label])
    return per_molecule_values, tick_labels, present_colors


def _find_values_for_fragment(
    analysis: dict,
    fragment: Fragment,
    parameter: ForceFieldParameter,
) -> list[float]:
    for frag, param_map in analysis.items():
        if frag.pattern == fragment.pattern:
            return param_map.get(parameter, [])
    return []


def _parameter_label(parameter: ForceFieldParameter) -> str:
    name = parameter.name.replace("_", " ").lower()
    if isinstance(parameter, BondParameter):
        return "Bond k (kcal/mol/Å²)" if "force" in name else "r₀ (Å)"
    if isinstance(parameter, AngleParameter):
        return "Angle k (kcal/mol/rad²)" if "force" in name else "θ₀ (deg)"
    if isinstance(parameter, DihedralParameter):
        return "φ_k (kcal/mol)" if "force" in name else ("Phase (rad)" if "phase" in name else "Periodicity")
    if isinstance(parameter, AtomParameter):
        if "charge" in name:
            return "Charge (e)"
        if "epsilon" in name:
            return "ε (kcal/mol)"
        if "sigma" in name:
            return "σ (Å)"
        return "Mass (amu)"
    return name
