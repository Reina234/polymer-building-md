from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.utils.parmed_helper import CoordinateCrosswalk
from polymer_md.visualisation._mol2d import embed_mol_in_ax, prepare_mol_2d, render_mol_2d
from polymer_md.visualisation._palette import REGION_FACE, REGION_HEX, REGION_RGB, clean_ax

_REGION_ORDER = ["left", "central", "right", "cap"]
_REGION_LABELS = {"left": "Left", "central": "Central", "right": "Right", "cap": "Cap"}

_FIGURE_SIZE = (17, 7)
_WIDTH_RATIOS = [1.1, 1]
_GRID_WSPACE = 0.06
_TITLE_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 9
_ANNOTATION_FONT_SIZE = 9.5
_AXIS_LABEL_FONT_SIZE = 10
_REGION_LABEL_FONT_SIZE = 7.5


def plot_parameterised_trimer(pt: ParameterisedTrimer) -> plt.Figure:
    crosswalk = CoordinateCrosswalk.map_mol3d_to_parmed(pt.mol_3d, pt.structure)
    mol_2d, mol3d_to_2d = prepare_mol_2d(pt.mol_3d)

    region_membership = _build_region_membership(pt)
    atom_rgb_map = _build_atom_rgb_map(mol3d_to_2d, region_membership)
    atom_notes = _build_atom_notes(mol3d_to_2d, crosswalk, pt)

    heavy_atoms = _collect_heavy_atom_data(pt, crosswalk, region_membership)

    fig = plt.figure(figsize=_FIGURE_SIZE, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=_WIDTH_RATIOS, wspace=_GRID_WSPACE)

    ax_mol = fig.add_subplot(gs[0])
    img = render_mol_2d(mol_2d, atom_rgb_map, atom_notes)
    embed_mol_in_ax(ax_mol, img)
    legend_handles = [
        mpatches.Patch(facecolor=REGION_HEX[r], edgecolor="#555", label=_REGION_LABELS[r], linewidth=0.8)
        for r in _REGION_ORDER
        if any(d["region"] == r for d in heavy_atoms)
    ]
    ax_mol.legend(handles=legend_handles, loc="lower right", fontsize=_LEGEND_FONT_SIZE, framealpha=0.9)

    ax_charge = fig.add_subplot(gs[1])
    _draw_charge_panel(ax_charge, heavy_atoms)

    fig.suptitle(
        f"Parameterised Trimer — {pt.trimer_result.label}  (p = {pt.trimer_result.probability:.3f})",
        fontsize=_TITLE_FONT_SIZE,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _build_region_membership(pt: ParameterisedTrimer) -> dict[int, str]:
    membership: dict[int, str] = {}
    for idx in pt.trimer_result.left_atom_indices:
        membership[idx] = "left"
    for idx in pt.trimer_result.central_atom_indices:
        membership[idx] = "central"
    for idx in pt.trimer_result.right_atom_indices:
        membership[idx] = "right"
    for idx in pt.trimer_result.cap_atom_indices:
        membership[idx] = "cap"
    return membership


def _build_atom_rgb_map(
    mol3d_to_2d: dict[int, int],
    region_membership: dict[int, str],
) -> dict[int, tuple[float, float, float]]:
    result: dict[int, tuple[float, float, float]] = {}
    for mol3d_idx, mol2d_idx in mol3d_to_2d.items():
        region = region_membership.get(mol3d_idx, "cap")
        result[mol2d_idx] = REGION_RGB[region]
    return result


def _build_atom_notes(
    mol3d_to_2d: dict[int, int],
    crosswalk: dict[int, int],
    pt: ParameterisedTrimer,
) -> dict[int, str]:
    notes: dict[int, str] = {}
    for mol3d_idx, mol2d_idx in mol3d_to_2d.items():
        parmed_idx = crosswalk.get(mol3d_idx)
        if parmed_idx is not None:
            gaff2_type = pt.structure.atoms[parmed_idx].type
            if gaff2_type:
                notes[mol2d_idx] = str(gaff2_type)
    return notes


def _collect_heavy_atom_data(
    pt: ParameterisedTrimer,
    crosswalk: dict[int, int],
    region_membership: dict[int, str],
) -> list[dict]:
    region_order = {r: i for i, r in enumerate(_REGION_ORDER)}
    atoms = []
    for mol3d_idx, parmed_idx in crosswalk.items():
        atom = pt.mol_3d.GetAtomWithIdx(mol3d_idx)
        if atom.GetAtomicNum() == 1:
            continue
        region = region_membership.get(mol3d_idx, "cap")
        parmed_atom = pt.structure.atoms[parmed_idx]
        atoms.append({
            "mol3d_idx": mol3d_idx,
            "region": region,
            "symbol": atom.GetSymbol(),
            "charge": parmed_atom.charge,
            "gaff2_type": parmed_atom.type or "",
        })
    atoms.sort(key=lambda d: (region_order[d["region"]], d["mol3d_idx"]))
    return atoms


def _draw_charge_panel(ax: plt.Axes, heavy_atoms: list[dict]) -> None:
    clean_ax(ax)

    charges = np.array([d["charge"] for d in heavy_atoms])
    regions = [d["region"] for d in heavy_atoms]
    bar_colors = [REGION_HEX[r] for r in regions]
    x = np.arange(len(heavy_atoms))

    ax.bar(x, charges, color=bar_colors, width=0.75, zorder=3, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#333333", linewidth=0.9, zorder=4)

    _add_region_shading(ax, heavy_atoms, x)

    total = float(np.sum(charges))
    ax.text(
        0.98, 0.97,
        f"Q$_{{total}}$ = {total:+.4f} e",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=_ANNOTATION_FONT_SIZE,
        bbox=dict(facecolor="white", edgecolor="#CCCCCC", boxstyle="round,pad=0.3", alpha=0.85),
    )

    ax.set_xticks([])
    ax.set_ylabel("Partial charge (e)", fontsize=_AXIS_LABEL_FONT_SIZE)
    ax.set_title("Per-atom partial charges", fontsize=_AXIS_LABEL_FONT_SIZE, pad=6)
    ax.grid(axis="y", color="#EBEBEB", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.7, len(heavy_atoms) - 0.3)


def _add_region_shading(ax: plt.Axes, heavy_atoms: list[dict], x: np.ndarray) -> None:
    if not heavy_atoms:
        return
    current_region = heavy_atoms[0]["region"]
    start = 0

    def _shade(start_x: int, end_x: int, region: str) -> None:
        ax.axvspan(
            start_x - 0.5, end_x - 0.5,
            alpha=0.12,
            color=REGION_HEX[region],
            zorder=0,
        )
        mid = (start_x + end_x - 1) / 2
        ax.text(
            mid, 1.01,
            _REGION_LABELS[region],
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=_REGION_LABEL_FONT_SIZE,
            color=REGION_HEX[region],
            fontweight="bold",
        )

    for i, atom in enumerate(heavy_atoms[1:], start=1):
        if atom["region"] != current_region:
            _shade(start, i, current_region)
            current_region = atom["region"]
            start = i
    _shade(start, len(heavy_atoms), current_region)
