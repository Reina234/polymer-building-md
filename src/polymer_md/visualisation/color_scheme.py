from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.visualisation._palette import COMPARISON_PALETTE, CHARGE_CMAP

LegendEntry = tuple[str, str]  # (label, hex_color)


@runtime_checkable
class ColorScheme(Protocol):
    @property
    def name(self) -> str: ...

    def atom_color(self, atom_idx: int, molecule: ParameterisedMolecule) -> str: ...

    def legend(self, molecule: ParameterisedMolecule) -> list[LegendEntry]: ...


# ---------------------------------------------------------------------------
# Charge
# ---------------------------------------------------------------------------

@dataclass
class ChargeColorScheme:
    n_legend_steps: int = 5

    @property
    def name(self) -> str:
        return "Charge"

    def atom_color(self, atom_idx: int, molecule: ParameterisedMolecule) -> str:
        charges = np.array([a.charge for a in molecule.structure.atoms])
        return _charge_hex(molecule.structure.atoms[atom_idx].charge, charges)

    def legend(self, molecule: ParameterisedMolecule) -> list[LegendEntry]:
        charges = np.array([a.charge for a in molecule.structure.atoms])
        vmax = max(float(np.abs(charges).max()), 0.01)
        steps = np.linspace(-vmax, vmax, self.n_legend_steps)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap_obj = matplotlib.colormaps[CHARGE_CMAP]
        return [(f"{v:+.3f} e", mcolors.to_hex(cmap_obj(norm(v)))) for v in steps]


def _charge_hex(charge: float, all_charges: np.ndarray) -> str:
    vmax = max(float(np.abs(all_charges).max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return mcolors.to_hex(matplotlib.colormaps[CHARGE_CMAP](norm(charge)))


# ---------------------------------------------------------------------------
# Residue identity
# ---------------------------------------------------------------------------

@dataclass
class ResidueColorScheme:
    _residue_colors: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return "Residue"

    def atom_color(self, atom_idx: int, molecule: ParameterisedMolecule) -> str:
        if not molecule.atom_metadata:
            return "#AAAAAA"
        residue_id = molecule.atom_metadata.get(atom_idx, (None, None))[0]
        if residue_id is None:
            return "#DDDDDD"
        color_map = self._build_color_map(molecule)
        return color_map.get(residue_id, "#AAAAAA")

    def legend(self, molecule: ParameterisedMolecule) -> list[LegendEntry]:
        color_map = self._build_color_map(molecule)
        return [(residue_id, color) for residue_id, color in color_map.items()]

    def _build_color_map(self, molecule: ParameterisedMolecule) -> dict[str, str]:
        residue_ids = _unique_residue_ids(molecule)
        return {
            residue_id: COMPARISON_PALETTE[i % len(COMPARISON_PALETTE)]
            for i, residue_id in enumerate(sorted(residue_ids))
        }


def _unique_residue_ids(molecule: ParameterisedMolecule) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for residue_id, _ in molecule.atom_metadata.values():
        if residue_id not in seen:
            seen.add(residue_id)
            ordered.append(residue_id)
    return ordered


# ---------------------------------------------------------------------------
# Element (CPK)
# ---------------------------------------------------------------------------

_CPK_HEX: dict[int, str] = {
    1: "#FFFFFF",
    6: "#909090",
    7: "#3050F8",
    8: "#FF0D0D",
    9: "#90E050",
    15: "#FF8000",
    16: "#FFFF30",
    17: "#1FF01F",
    35: "#A62929",
    53: "#940094",
}
_CPK_DEFAULT = "#BEA06E"

_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


@dataclass
class ElementColorScheme:
    @property
    def name(self) -> str:
        return "Element"

    def atom_color(self, atom_idx: int, molecule: ParameterisedMolecule) -> str:
        atomic_number = molecule.structure.atoms[atom_idx].atomic_number
        return _CPK_HEX.get(atomic_number, _CPK_DEFAULT)

    def legend(self, molecule: ParameterisedMolecule) -> list[LegendEntry]:
        present = {a.atomic_number for a in molecule.structure.atoms}
        return [
            (_ELEMENT_SYMBOLS.get(z, str(z)), _CPK_HEX.get(z, _CPK_DEFAULT))
            for z in sorted(present)
        ]


# ---------------------------------------------------------------------------
# GAFF2 atom type
# ---------------------------------------------------------------------------

_TYPE_PREFIX_HEX: dict[str, str] = {
    "c": "#606060",
    "n": "#3050F8",
    "o": "#FF0D0D",
    "s": "#D4C000",
    "p": "#FF8000",
    "f": "#90E050",
    "cl": "#1FF01F",
    "br": "#A62929",
    "i": "#940094",
    "h": "#E0E0E0",
}
_TYPE_DEFAULT_HEX = "#BEA06E"


@dataclass
class AtomTypeColorScheme:
    @property
    def name(self) -> str:
        return "GAFF2 Type"

    def atom_color(self, atom_idx: int, molecule: ParameterisedMolecule) -> str:
        gaff2_type = _gaff2_type(molecule.structure.atoms[atom_idx])
        return _type_hex(gaff2_type)

    def legend(self, molecule: ParameterisedMolecule) -> list[LegendEntry]:
        seen: dict[str, str] = {}
        for atom in molecule.structure.atoms:
            gaff2_type = _gaff2_type(atom)
            if gaff2_type not in seen:
                seen[gaff2_type] = _type_hex(gaff2_type)
        return sorted(seen.items())


def _gaff2_type(atom) -> str:
    atom_type = getattr(atom, "atom_type", None)
    if atom_type is not None and hasattr(atom_type, "name") and atom_type.name:
        return atom_type.name
    return atom.name or "?"


def _type_hex(gaff2_type: str) -> str:
    prefix = gaff2_type.lower()
    for key in sorted(_TYPE_PREFIX_HEX, key=len, reverse=True):
        if prefix.startswith(key):
            return _TYPE_PREFIX_HEX[key]
    return _TYPE_DEFAULT_HEX
