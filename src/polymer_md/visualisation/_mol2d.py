from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

if TYPE_CHECKING:
    from PIL import Image as PILImage


def prepare_mol_2d(mol_3d: Chem.Mol) -> tuple[Chem.Mol, dict[int, int]]:
    rw = Chem.RWMol(mol_3d)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    mol_2d = Chem.RemoveAllHs(rw.GetMol())
    AllChem.Compute2DCoords(mol_2d)
    mol3d_to_2d: dict[int, int] = {}
    for atom in mol_2d.GetAtoms():
        mol3d_idx = atom.GetAtomMapNum() - 1
        mol3d_to_2d[mol3d_idx] = atom.GetIdx()
        atom.SetAtomMapNum(0)
    return mol_2d, mol3d_to_2d


def render_mol_2d(
    mol_2d: Chem.Mol,
    atom_rgb_map: dict[int, tuple[float, float, float]],
    atom_notes: dict[int, str] | None = None,
    width: int = 700,
    height: int = 500,
) -> "PILImage":
    from PIL import Image

    mol = Chem.RWMol(mol_2d)
    if atom_notes:
        for idx, note in atom_notes.items():
            mol.GetAtomWithIdx(idx).SetProp("atomNote", note)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.padding = 0.12
    opts.atomHighlightsAreCircles = True
    opts.annotationFontScale = 0.55

    all_indices = list(range(mol.GetNumAtoms()))
    default_rgb = (0.92, 0.92, 0.92)
    highlight_colors = {i: atom_rgb_map.get(i, default_rgb) for i in all_indices}
    highlight_radii = {i: 0.45 for i in all_indices}

    bond_colors: dict[int, tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ci = atom_rgb_map.get(i, default_rgb)
        cj = atom_rgb_map.get(j, default_rgb)
        bond_colors[bond.GetIdx()] = ci if ci == cj else (0.72, 0.72, 0.72)

    drawer.DrawMolecule(
        mol.GetMol(),
        highlightAtoms=all_indices,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def embed_mol_in_ax(ax: plt.Axes, img: "PILImage") -> None:
    ax.imshow(img, aspect="equal")
    ax.set_axis_off()
