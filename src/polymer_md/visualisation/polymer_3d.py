from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.utils.parmed_helper import CoordinateCrosswalk
from polymer_md.visualisation._3d_base import display_3d, structure_to_pdb_string
from polymer_md.visualisation._palette import CHARGE_CMAP


def view_polymer_3d(pm: ParameterisedMolecule) -> object:
    import py3Dmol  # type: ignore[import-untyped]

    pdb_str = structure_to_pdb_string(pm.structure)
    charges = np.array([atom.charge for atom in pm.structure.atoms])

    vmax = max(abs(charges).max(), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_obj = cm.get_cmap(CHARGE_CMAP)

    view = py3Dmol.view(width=820, height=540)
    view.addModel(pdb_str, "pdb")

    view.setStyle({}, {"stick": {"colorscheme": "greyCarbon", "radius": 0.08}})

    for atom in pm.structure.atoms:
        rgba = cmap_obj(norm(atom.charge))
        hex_color = mcolors.to_hex(rgba)
        radius = 0.28 if atom.atomic_number != 1 else 0.14
        view.addStyle(
            {"serial": [atom.number]},
            {
                "sphere": {"color": hex_color, "radius": radius},
                "stick": {"color": hex_color, "radius": 0.08},
            },
        )

    view.setBackgroundColor("#FAFAFA")
    view.zoomTo()
    _add_charge_legend(view, vmax, norm, cmap_obj)

    return display_3d(view)


def _add_charge_legend(
    view: object,
    vmax: float,
    norm: mcolors.Normalize,
    cmap_obj,
) -> None:
    legend_values = [-vmax, -vmax / 2, 0.0, vmax / 2, vmax]
    for i, charge_val in enumerate(legend_values):
        rgba = cmap_obj(norm(charge_val))
        hex_color = mcolors.to_hex(rgba)
        label_text = f"{charge_val:+.3f} e"
        view.addLabel(  # type: ignore[attr-defined]
            label_text,
            {
                "position": {"x": -100, "y": 10 - i * 8, "z": 0},
                "backgroundColor": hex_color,
                "fontColor": "white" if charge_val != 0.0 else "#333333",
                "fontSize": 10,
                "backgroundOpacity": 0.85,
                "useScreen": True,
            },
        )
