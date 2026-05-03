from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.visualisation.color_scheme import (
    AtomTypeColorScheme,
    ChargeColorScheme,
    ColorScheme,
    ElementColorScheme,
    ResidueColorScheme,
)
from polymer_md.visualisation.polymer_3d import _legend_font_color


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_atom(idx: int, number: int, atomic_number: int, charge: float, name: str = "c3") -> MagicMock:
    atom = MagicMock()
    atom.idx = idx
    atom.number = number
    atom.atomic_number = atomic_number
    atom.charge = charge
    atom.name = name
    atom.atom_type = MagicMock()
    atom.atom_type.name = name
    return atom


def _make_molecule(
    atoms: list[MagicMock],
    atom_metadata: dict[int, tuple[str, int]] | None = None,
) -> ParameterisedMolecule:
    structure = MagicMock()
    structure.atoms = atoms
    mol = MagicMock()
    source = MagicMock()
    return ParameterisedMolecule(
        structure=structure,
        mol=mol,
        source=source,
        atom_metadata=atom_metadata or {},
    )


def _homopolymer_molecule() -> ParameterisedMolecule:
    atoms = [
        _mock_atom(0, 1, 6, -0.1, "c3"),
        _mock_atom(1, 2, 6, 0.2, "ca"),
        _mock_atom(2, 3, 1, 0.05, "hc"),
    ]
    return _make_molecule(atoms, {0: ("MMA", 0), 1: ("MMA", 1)})


def _copolymer_molecule() -> ParameterisedMolecule:
    atoms = [
        _mock_atom(0, 1, 6, -0.1, "c3"),
        _mock_atom(1, 2, 6, 0.2, "ca"),
        _mock_atom(2, 3, 8, -0.3, "oh"),
    ]
    return _make_molecule(atoms, {0: ("MMA", 0), 1: ("BA", 0), 2: ("BA", 1)})


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestColorSchemeProtocol:
    @pytest.mark.parametrize("scheme_cls", [
        ChargeColorScheme, ResidueColorScheme, ElementColorScheme, AtomTypeColorScheme
    ])
    def test_implements_color_scheme_protocol(self, scheme_cls):
        assert isinstance(scheme_cls(), ColorScheme)

    @pytest.mark.parametrize("scheme_cls", [
        ChargeColorScheme, ResidueColorScheme, ElementColorScheme, AtomTypeColorScheme
    ])
    def test_name_is_nonempty_string(self, scheme_cls):
        assert isinstance(scheme_cls().name, str)
        assert len(scheme_cls().name) > 0

    @pytest.mark.parametrize("scheme_cls", [
        ChargeColorScheme, ResidueColorScheme, ElementColorScheme, AtomTypeColorScheme
    ])
    def test_atom_color_returns_hex_string(self, scheme_cls):
        mol = _homopolymer_molecule()
        color = scheme_cls().atom_color(0, mol)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7

    @pytest.mark.parametrize("scheme_cls", [
        ChargeColorScheme, ResidueColorScheme, ElementColorScheme, AtomTypeColorScheme
    ])
    def test_legend_returns_list_of_pairs(self, scheme_cls):
        mol = _homopolymer_molecule()
        entries = scheme_cls().legend(mol)
        assert isinstance(entries, list)
        for label, color in entries:
            assert isinstance(label, str)
            assert color.startswith("#")


# ---------------------------------------------------------------------------
# ChargeColorScheme
# ---------------------------------------------------------------------------

class TestChargeColorScheme:
    def test_positive_charge_maps_to_warm_color(self):
        atoms = [_mock_atom(0, 1, 6, 1.0), _mock_atom(1, 2, 6, -1.0)]
        mol = _make_molecule(atoms)
        positive_color = ChargeColorScheme().atom_color(0, mol)
        negative_color = ChargeColorScheme().atom_color(1, mol)
        assert positive_color != negative_color

    def test_zero_charge_maps_to_lighter_color_than_extremes(self):
        atoms = [_mock_atom(0, 1, 6, 0.0), _mock_atom(1, 2, 6, -1.0), _mock_atom(2, 3, 6, 1.0)]
        mol = _make_molecule(atoms)
        import matplotlib.colors as mcolors
        zero_color = ChargeColorScheme().atom_color(0, mol)
        negative_color = ChargeColorScheme().atom_color(1, mol)
        positive_color = ChargeColorScheme().atom_color(2, mol)
        zero_rgb = mcolors.to_rgb(zero_color)
        neg_rgb = mcolors.to_rgb(negative_color)
        pos_rgb = mcolors.to_rgb(positive_color)
        zero_luminance = 0.299 * zero_rgb[0] + 0.587 * zero_rgb[1] + 0.114 * zero_rgb[2]
        neg_luminance = 0.299 * neg_rgb[0] + 0.587 * neg_rgb[1] + 0.114 * neg_rgb[2]
        pos_luminance = 0.299 * pos_rgb[0] + 0.587 * pos_rgb[1] + 0.114 * pos_rgb[2]
        assert zero_luminance > neg_luminance or zero_luminance > pos_luminance

    def test_legend_has_n_steps(self):
        mol = _homopolymer_molecule()
        legend = ChargeColorScheme(n_legend_steps=7).legend(mol)
        assert len(legend) == 7

    def test_legend_labels_contain_charge_unit(self):
        mol = _homopolymer_molecule()
        legend = ChargeColorScheme().legend(mol)
        for label, _ in legend:
            assert "e" in label


# ---------------------------------------------------------------------------
# ResidueColorScheme
# ---------------------------------------------------------------------------

class TestResidueColorScheme:
    def test_different_residues_get_different_colors(self):
        mol = _copolymer_molecule()
        scheme = ResidueColorScheme()
        color_mma = scheme.atom_color(0, mol)
        color_ba = scheme.atom_color(1, mol)
        assert color_mma != color_ba

    def test_same_residue_atoms_get_same_color(self):
        atoms = [_mock_atom(0, 1, 6, 0.0), _mock_atom(1, 2, 6, 0.0)]
        mol = _make_molecule(atoms, {0: ("MMA", 0), 1: ("MMA", 1)})
        scheme = ResidueColorScheme()
        assert scheme.atom_color(0, mol) == scheme.atom_color(1, mol)

    def test_legend_contains_all_residue_ids(self):
        mol = _copolymer_molecule()
        legend = ResidueColorScheme().legend(mol)
        labels = [label for label, _ in legend]
        assert "MMA" in labels
        assert "BA" in labels

    def test_empty_metadata_returns_fallback_color(self):
        atoms = [_mock_atom(0, 1, 6, 0.0)]
        mol = _make_molecule(atoms, atom_metadata={})
        color = ResidueColorScheme().atom_color(0, mol)
        assert color.startswith("#")

    def test_atom_not_in_metadata_returns_fallback(self):
        atoms = [_mock_atom(0, 1, 6, 0.0), _mock_atom(1, 2, 6, 0.0)]
        mol = _make_molecule(atoms, {0: ("MMA", 0)})
        color = ResidueColorScheme().atom_color(1, mol)
        assert color.startswith("#")


# ---------------------------------------------------------------------------
# ElementColorScheme
# ---------------------------------------------------------------------------

class TestElementColorScheme:
    def test_carbon_is_grey(self):
        atoms = [_mock_atom(0, 1, 6, 0.0)]
        mol = _make_molecule(atoms)
        color = ElementColorScheme().atom_color(0, mol)
        assert color.lower() == "#909090"

    def test_oxygen_is_red(self):
        atoms = [_mock_atom(0, 1, 8, 0.0)]
        mol = _make_molecule(atoms)
        color = ElementColorScheme().atom_color(0, mol)
        assert color.lower() == "#ff0d0d"

    def test_hydrogen_is_white(self):
        atoms = [_mock_atom(0, 1, 1, 0.0)]
        mol = _make_molecule(atoms)
        color = ElementColorScheme().atom_color(0, mol)
        assert color.lower() == "#ffffff"

    def test_unknown_element_returns_fallback(self):
        atoms = [_mock_atom(0, 1, 99, 0.0)]
        mol = _make_molecule(atoms)
        color = ElementColorScheme().atom_color(0, mol)
        assert color.startswith("#")

    def test_legend_includes_all_present_elements(self):
        atoms = [_mock_atom(0, 1, 6, 0.0), _mock_atom(1, 2, 8, 0.0)]
        mol = _make_molecule(atoms)
        legend = ElementColorScheme().legend(mol)
        labels = [label for label, _ in legend]
        assert "C" in labels
        assert "O" in labels


# ---------------------------------------------------------------------------
# AtomTypeColorScheme
# ---------------------------------------------------------------------------

class TestAtomTypeColorScheme:
    def test_carbon_types_return_dark_color(self):
        atoms = [_mock_atom(0, 1, 6, 0.0, "c3")]
        mol = _make_molecule(atoms)
        color = AtomTypeColorScheme().atom_color(0, mol)
        assert color.startswith("#")

    def test_oxygen_types_return_red_family(self):
        atoms = [_mock_atom(0, 1, 8, 0.0, "oh")]
        mol = _make_molecule(atoms)
        color = AtomTypeColorScheme().atom_color(0, mol)
        assert color.lower() == "#ff0d0d"

    def test_legend_contains_type_names(self):
        atoms = [_mock_atom(0, 1, 6, 0.0, "c3"), _mock_atom(1, 2, 8, 0.0, "oh")]
        mol = _make_molecule(atoms)
        legend = AtomTypeColorScheme().legend(mol)
        labels = [label for label, _ in legend]
        assert "c3" in labels
        assert "oh" in labels

    def test_distinct_types_get_same_family_color(self):
        atoms = [_mock_atom(0, 1, 6, 0.0, "c3"), _mock_atom(1, 2, 6, 0.0, "ca")]
        mol = _make_molecule(atoms)
        scheme = AtomTypeColorScheme()
        assert scheme.atom_color(0, mol) == scheme.atom_color(1, mol)


# ---------------------------------------------------------------------------
# Legend font color
# ---------------------------------------------------------------------------

class TestLegendFontColor:
    def test_dark_background_gives_white_text(self):
        assert _legend_font_color("#000000") == "white"

    def test_light_background_gives_dark_text(self):
        assert _legend_font_color("#FFFFFF") == "#333333"
