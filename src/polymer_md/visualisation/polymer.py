from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.utils.parmed_helper import CoordinateCrosswalk
from polymer_md.visualisation._mol2d import embed_mol_in_ax, prepare_mol_2d, render_mol_2d
from polymer_md.visualisation._palette import CHARGE_CMAP, clean_ax


def plot_parameterised_polymer(pm: ParameterisedMolecule) -> plt.Figure:
    crosswalk = CoordinateCrosswalk.map_mol3d_to_parmed(pm.mol, pm.structure)
    mol_2d, mol3d_to_2d = prepare_mol_2d(pm.mol)

    charges = np.array([pm.structure.atoms[i].charge for i in range(len(pm.structure.atoms))])
    heavy_charges = _heavy_atom_charges(pm, crosswalk)

    norm, cmap_obj = _charge_norm_and_cmap(heavy_charges)
    atom_rgb_map = _build_atom_rgb_map(mol3d_to_2d, crosswalk, pm, norm, cmap_obj)

    fig = plt.figure(figsize=(17, 7), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.06)

    ax_mol = fig.add_subplot(gs[0])
    img = render_mol_2d(mol_2d, atom_rgb_map)
    embed_mol_in_ax(ax_mol, img)
    _add_colorbar(fig, ax_mol, norm, cmap_obj)

    right_gs = gs[1].subgridspec(2, 1, hspace=0.45, height_ratios=[3, 2])
    ax_hist = fig.add_subplot(right_gs[0])
    ax_stats = fig.add_subplot(right_gs[1])

    _draw_charge_histogram(ax_hist, heavy_charges, norm, cmap_obj)
    _draw_stats(ax_stats, pm, charges, heavy_charges)

    fig.suptitle("Parameterised Polymer", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def _heavy_atom_charges(pm: ParameterisedMolecule, crosswalk: dict[int, int]) -> np.ndarray:
    from rdkit import Chem
    return np.array([
        pm.structure.atoms[crosswalk[i]].charge
        for i in range(pm.mol.GetNumAtoms())
        if pm.mol.GetAtomWithIdx(i).GetAtomicNum() != 1 and i in crosswalk
    ])


def _charge_norm_and_cmap(
    charges: np.ndarray,
) -> tuple[mcolors.Normalize, cm.ScalarMappable]:
    vmax = max(abs(charges).max(), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_obj = plt.get_cmap(CHARGE_CMAP)
    return norm, cmap_obj


def _build_atom_rgb_map(
    mol3d_to_2d: dict[int, int],
    crosswalk: dict[int, int],
    pm: ParameterisedMolecule,
    norm: mcolors.Normalize,
    cmap_obj,
) -> dict[int, tuple[float, float, float]]:
    result: dict[int, tuple[float, float, float]] = {}
    for mol3d_idx, mol2d_idx in mol3d_to_2d.items():
        parmed_idx = crosswalk.get(mol3d_idx)
        if parmed_idx is not None:
            charge = pm.structure.atoms[parmed_idx].charge
            rgba = cmap_obj(norm(charge))
            result[mol2d_idx] = rgba[:3]
    return result


def _add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    norm: mcolors.Normalize,
    cmap_obj,
) -> None:
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.01)
    cbar.set_label("Partial charge (e)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)


def _draw_charge_histogram(
    ax: plt.Axes,
    heavy_charges: np.ndarray,
    norm: mcolors.Normalize,
    cmap_obj,
) -> None:
    clean_ax(ax)
    n_bins = min(30, max(10, len(heavy_charges) // 3))
    counts, bin_edges = np.histogram(heavy_charges, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_colors = [cmap_obj(norm(c)) for c in bin_centers]
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color=bar_colors,
           align="edge", edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=4)
    ax.set_xlabel("Partial charge (e)", fontsize=9.5)
    ax.set_ylabel("Count", fontsize=9.5)
    ax.set_title("Heavy-atom charge distribution", fontsize=9.5, pad=5)
    ax.grid(axis="y", color="#EBEBEB", linewidth=0.8, zorder=0)


def _draw_stats(
    ax: plt.Axes,
    pm: ParameterisedMolecule,
    all_charges: np.ndarray,
    heavy_charges: np.ndarray,
) -> None:
    ax.set_axis_off()
    total = float(all_charges.sum())
    n_total = len(pm.structure.atoms)
    n_heavy = len(heavy_charges)
    n_residues = len(pm.structure.residues) if hasattr(pm.structure, "residues") else "—"
    charge_range = f"[{heavy_charges.min():+.3f}, {heavy_charges.max():+.3f}]"
    std = float(heavy_charges.std())

    lines = [
        ("Total atoms", f"{n_total}"),
        ("Heavy atoms", f"{n_heavy}"),
        ("Residues", f"{n_residues}"),
        ("Total charge", f"{total:+.6f} e"),
        ("Charge range", charge_range),
        ("Charge std", f"{std:.4f} e"),
    ]
    col_x = [0.02, 0.62]
    row_y = 0.88
    row_step = 0.145
    ax.text(0.5, 1.0, "Summary", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, fontweight="bold")
    for i, (label, value) in enumerate(lines):
        y = row_y - i * row_step
        ax.text(col_x[0], y, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#555555")
        ax.text(col_x[1], y, value, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold")
