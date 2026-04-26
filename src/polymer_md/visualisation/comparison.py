from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from polymer_md.analysis.results import ComparisonResult
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import ForceFieldParameter
from polymer_md.visualisation._palette import COMPARISON_PALETTE, clean_ax

_MAX_COLS = 3


def plot_comparison(result: ComparisonResult) -> plt.Figure:
    pairs = _collect_pairs(result)
    if not pairs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    n_cols = min(_MAX_COLS, len(pairs))
    n_rows = math.ceil(len(pairs) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
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

    fig.suptitle("Parameter Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def _collect_pairs(result: ComparisonResult) -> list[tuple[Fragment, ForceFieldParameter]]:
    seen: set[tuple] = set()
    ordered: list[tuple[Fragment, ForceFieldParameter]] = []
    for analysis in result.results.values():
        for fragment, param_map in analysis.items():
            for parameter, values in param_map.items():
                if values:
                    key = (fragment.pattern, type(parameter).__name__, parameter)
                    if key not in seen:
                        seen.add(key)
                        ordered.append((fragment, parameter))
    return ordered


def _draw_comparison_cell(
    ax: plt.Axes,
    result: ComparisonResult,
    fragment: Fragment,
    parameter: ForceFieldParameter,
    molecule_labels: list[str],
    color_map: dict[str, str],
) -> None:
    clean_ax(ax)

    per_molecule_values = []
    tick_labels = []
    present_colors = []
    for label in molecule_labels:
        analysis = result.results.get(label, {})
        values = []
        for frag, param_map in analysis.items():
            if frag.pattern == fragment.pattern:
                values = param_map.get(parameter, [])
                break
        if values:
            per_molecule_values.append(values)
            tick_labels.append(label)
            present_colors.append(color_map[label])

    if not per_molecule_values:
        ax.set_axis_off()
        return

    positions = np.arange(1, len(per_molecule_values) + 1)
    bp = ax.boxplot(
        per_molecule_values,
        positions=positions,
        widths=0.5,
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
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color=color,
            s=18,
            alpha=0.7,
            zorder=5,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(_parameter_label(parameter), fontsize=9)
    ax.set_title(f"{fragment.pattern}\n{parameter.name.replace('_', ' ').title()}", fontsize=9, pad=5)
    ax.grid(axis="y", color="#EBEBEB", linewidth=0.8, zorder=0)


def _parameter_label(parameter: ForceFieldParameter) -> str:
    name = parameter.name.replace("_", " ").lower()
    from polymer_md.parameterisation.fragments.data_models.parameters import (
        AngleParameter, BondParameter, DihedralParameter, AtomParameter,
    )
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
