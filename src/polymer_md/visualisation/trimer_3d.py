from __future__ import annotations

from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.utils.parmed_helper import CoordinateCrosswalk
from polymer_md.visualisation._3d_base import display_3d, structure_to_pdb_string
from polymer_md.visualisation._palette import REGION_HEX

_REGION_ORDER = [
    ("left", "left_atom_indices"),
    ("central", "central_atom_indices"),
    ("right", "right_atom_indices"),
    ("cap", "cap_atom_indices"),
]


def view_trimer_3d(pt: ParameterisedTrimer) -> object:
    import py3Dmol  # type: ignore[import-untyped]

    crosswalk = CoordinateCrosswalk.map_mol3d_to_parmed(pt.mol_3d, pt.structure)
    pdb_str = structure_to_pdb_string(pt.structure)

    view = py3Dmol.view(width=820, height=540)
    view.addModel(pdb_str, "pdb")

    view.setStyle({}, {"stick": {"colorscheme": "greyCarbon", "radius": 0.10}})
    view.addStyle({"elem": "H"}, {"sphere": {"radius": 0.12, "color": "#E0E0E0"}})

    for region_name, indices_attr in _REGION_ORDER:
        mol3d_indices = getattr(pt.trimer_result, indices_attr)
        serials = _mol3d_to_serials(mol3d_indices, crosswalk, pt)
        if not serials:
            continue
        color = REGION_HEX[region_name]
        view.addStyle(
            {"serial": serials},
            {
                "sphere": {"color": color, "radius": 0.32},
                "stick": {"color": color, "radius": 0.14},
            },
        )

    view.setBackgroundColor("#FAFAFA")
    view.zoomTo()

    _add_legend_labels(view, pt, crosswalk)

    return display_3d(view)


def _mol3d_to_serials(
    mol3d_indices: frozenset[int],
    crosswalk: dict[int, int],
    pt: ParameterisedTrimer,
) -> list[int]:
    serials = []
    for mol3d_idx in mol3d_indices:
        parmed_idx = crosswalk.get(mol3d_idx)
        if parmed_idx is not None:
            serials.append(pt.structure.atoms[parmed_idx].number)
            for nbr_atom in pt.mol_3d.GetAtomWithIdx(mol3d_idx).GetNeighbors():
                if nbr_atom.GetAtomicNum() == 1:
                    h_parmed = crosswalk.get(nbr_atom.GetIdx())
                    if h_parmed is not None:
                        serials.append(pt.structure.atoms[h_parmed].number)
    return serials


def _add_legend_labels(
    view: object,
    pt: ParameterisedTrimer,
    crosswalk: dict[int, int],
) -> None:
    label_map = {
        "left": f"Left ({pt.trimer_result.left_id})",
        "central": f"Central ({pt.trimer_result.central_id})",
        "right": f"Right ({pt.trimer_result.right_id})",
        "cap": "Cap",
    }
    for region_name, indices_attr in _REGION_ORDER:
        mol3d_indices = getattr(pt.trimer_result, indices_attr)
        if not mol3d_indices:
            continue
        first_idx = next(iter(sorted(mol3d_indices)))
        parmed_idx = crosswalk.get(first_idx)
        if parmed_idx is None:
            continue
        atom = pt.structure.atoms[parmed_idx]
        view.addLabel(  # type: ignore[attr-defined]
            label_map[region_name],
            {
                "position": {"x": atom.xx, "y": atom.xy, "z": atom.xz},
                "backgroundColor": REGION_HEX[region_name],
                "fontColor": "white",
                "fontSize": 11,
                "backgroundOpacity": 0.8,
                "borderRadius": 3,
            },
        )
