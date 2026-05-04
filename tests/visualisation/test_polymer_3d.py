from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.visualisation.color_scheme import ChargeColorScheme, ElementColorScheme, ResidueColorScheme
from polymer_md.visualisation.polymer_3d import PolymerViewer, _wrap_panels_with_toggles


def _mock_atom(idx: int, atomic_number: int = 6, charge: float = 0.0, name: str = "c3") -> MagicMock:
    atom = MagicMock()
    atom.idx = idx
    atom.number = idx + 1
    atom.atomic_number = atomic_number
    atom.charge = charge
    atom.name = name
    atom.atom_type = MagicMock()
    atom.atom_type.name = name
    return atom


def _make_molecule(n_atoms: int = 3) -> ParameterisedMolecule:
    atoms = [_mock_atom(i, charge=float(i) * 0.1 - 0.1) for i in range(n_atoms)]
    structure = MagicMock()
    structure.atoms = atoms
    return ParameterisedMolecule(
        structure=structure,
        mol=MagicMock(),
        source=MagicMock(),
        atom_metadata={0: ("MMA", 0), 1: ("BA", 0), 2: ("BA", 1)},
    )


class TestPolymerViewerDefaults:
    def test_default_schemes_are_four(self):
        viewer = PolymerViewer(molecule=_make_molecule())
        assert len(viewer.schemes) == 4

    def test_accepts_custom_schemes(self):
        viewer = PolymerViewer(molecule=_make_molecule(), schemes=[ChargeColorScheme()])
        assert len(viewer.schemes) == 1

    def test_default_width_and_height(self):
        viewer = PolymerViewer(molecule=_make_molecule())
        assert viewer.width > 0
        assert viewer.height > 0


class TestPolymerViewerSave:
    def test_save_creates_html_file(self, tmp_path):
        viewer = PolymerViewer(
            molecule=_make_molecule(),
            schemes=[ChargeColorScheme(), ElementColorScheme()],
        )
        with patch("polymer_md.visualisation.polymer_3d.structure_to_pdb_string", return_value=""):
            with patch("py3Dmol.view") as mock_view:
                mock_view.return_value._make_html.return_value = "<html></html>"
                mock_view.return_value.addModel = MagicMock()
                mock_view.return_value.setStyle = MagicMock()
                mock_view.return_value.addStyle = MagicMock()
                mock_view.return_value.addLabel = MagicMock()
                mock_view.return_value.setBackgroundColor = MagicMock()
                mock_view.return_value.zoomTo = MagicMock()
                path = tmp_path / "output.html"
                viewer.save(path)
        assert path.exists()
        content = path.read_text()
        assert "<html>" in content.lower()

    def test_save_creates_parent_directories(self, tmp_path):
        viewer = PolymerViewer(molecule=_make_molecule(), schemes=[ChargeColorScheme()])
        with patch("polymer_md.visualisation.polymer_3d.structure_to_pdb_string", return_value=""):
            with patch("py3Dmol.view") as mock_view:
                mock_view.return_value._make_html.return_value = "<html></html>"
                mock_view.return_value.addModel = MagicMock()
                mock_view.return_value.setStyle = MagicMock()
                mock_view.return_value.addStyle = MagicMock()
                mock_view.return_value.addLabel = MagicMock()
                mock_view.return_value.setBackgroundColor = MagicMock()
                mock_view.return_value.zoomTo = MagicMock()
                nested = tmp_path / "deep" / "nested" / "view.html"
                viewer.save(nested)
        assert nested.exists()


class TestWrapPanelsWithToggles:
    def test_output_contains_all_scheme_names(self):
        html = _wrap_panels_with_toggles(
            ["<html>A</html>", "<html>B</html>"],
            ["Charge", "Residue"],
            width=800,
            height=500,
        )
        assert "Charge" in html
        assert "Residue" in html

    def test_output_contains_toggle_buttons(self):
        html = _wrap_panels_with_toggles(["<html></html>"], ["Charge"], 800, 500)
        assert "toggle-btn" in html

    def test_output_contains_iframes(self):
        html = _wrap_panels_with_toggles(["<html></html>", "<html></html>"], ["A", "B"], 800, 500)
        assert html.count("<iframe") == 2

    def test_first_panel_is_active(self):
        html = _wrap_panels_with_toggles(["<html></html>", "<html></html>"], ["A", "B"], 800, 500)
        assert 'class="toggle-btn active"' in html

    def test_embedded_content_is_base64(self):
        import base64
        inner = "<html>content</html>"
        html = _wrap_panels_with_toggles([inner], ["Test"], 800, 500)
        encoded = base64.b64encode(inner.encode()).decode()
        assert encoded in html

    def test_single_scheme_produces_one_iframe(self):
        html = _wrap_panels_with_toggles(["<html></html>"], ["Only"], 800, 500)
        assert html.count("<iframe") == 1

    def test_javascript_toggle_function_present(self):
        html = _wrap_panels_with_toggles(["<html></html>"], ["A"], 800, 500)
        assert "showPanel" in html


class TestParameterisedMoleculeAtomMetadata:
    def test_atom_metadata_stored_on_molecule(self):
        mol = _make_molecule()
        assert mol.atom_metadata[0] == ("MMA", 0)
        assert mol.atom_metadata[1] == ("BA", 0)

    def test_atom_metadata_defaults_to_empty(self):
        mol = ParameterisedMolecule(
            structure=MagicMock(), mol=MagicMock(), source=MagicMock()
        )
        assert mol.atom_metadata == {}
