from __future__ import annotations

import base64
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.visualisation._3d_base import is_jupyter, structure_to_pdb_string
from polymer_md.visualisation.color_scheme import (
    AtomTypeColorScheme,
    ChargeColorScheme,
    ColorScheme,
    ElementColorScheme,
    ResidueColorScheme,
)

_VIEWER_WIDTH = 820
_VIEWER_HEIGHT = 500
_SPHERE_RADIUS_HEAVY = 0.28
_SPHERE_RADIUS_HYDROGEN = 0.14
_STICK_RADIUS = 0.08
_LEGEND_X = -100
_LEGEND_Y_START = 10
_LEGEND_Y_STEP = -8
_LEGEND_FONT_SIZE = 10


def _default_schemes() -> list[ColorScheme]:
    return [ChargeColorScheme(), ResidueColorScheme(), ElementColorScheme(), AtomTypeColorScheme()]


@dataclass
class PolymerViewer:
    molecule: ParameterisedMolecule
    schemes: list[ColorScheme] = field(default_factory=_default_schemes)
    width: int = _VIEWER_WIDTH
    height: int = _VIEWER_HEIGHT

    def show(self, output_path: Path | None = None) -> None:
        if is_jupyter():
            self._show_jupyter()
        else:
            self._show_standalone(output_path)

    def save(self, output_path: Path) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(self._build_combined_html(), encoding="utf-8")

    # ------------------------------------------------------------------
    # Jupyter
    # ------------------------------------------------------------------

    def _show_jupyter(self) -> None:
        import ipywidgets as widgets
        from IPython.display import display

        if len(self.schemes) == 1:
            self._make_view(self.schemes[0]).show()
            return

        scheme_by_name = {s.name: s for s in self.schemes}
        output = widgets.Output()
        dropdown = widgets.Dropdown(
            options=[s.name for s in self.schemes],
            description="Coloring:",
            style={"description_width": "initial"},
        )

        def on_change(change: dict) -> None:
            output.clear_output(wait=True)
            with output:
                self._make_view(scheme_by_name[change["new"]]).show()

        with output:
            self._make_view(self.schemes[0]).show()

        dropdown.observe(on_change, names="value")
        display(widgets.VBox([dropdown, output]))

    # ------------------------------------------------------------------
    # Standalone HTML
    # ------------------------------------------------------------------

    def _show_standalone(self, output_path: Path | None = None) -> None:
        html = self._build_combined_html()
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(html, encoding="utf-8")
            path = str(output_path)
        else:
            with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(html)
                path = tmp.name
        webbrowser.open(f"file://{path}")

    def _build_combined_html(self) -> str:
        scheme_htmls = [self._scheme_to_html(scheme) for scheme in self.schemes]
        names = [s.name for s in self.schemes]
        return _wrap_panels_with_toggles(scheme_htmls, names, self.width, self.height)

    def _scheme_to_html(self, scheme: ColorScheme) -> str:
        fragment = self._make_view(scheme)._make_html()
        return (
            "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"></head>\n"
            "<body style=\"margin:0;padding:0;background:#FAFAFA;\">\n"
            f"{fragment}\n</body>\n</html>"
        )

    # ------------------------------------------------------------------
    # py3Dmol view construction
    # ------------------------------------------------------------------

    def _make_view(self, scheme: ColorScheme) -> object:
        import py3Dmol

        pdb_str = structure_to_pdb_string(self.molecule.structure)
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(pdb_str, "pdb")
        view.setStyle({}, {"stick": {"colorscheme": "greyCarbon", "radius": _STICK_RADIUS}})
        self._apply_atom_colors(view, scheme)
        self._add_legend(view, scheme)
        view.setBackgroundColor("#FAFAFA")
        view.zoomTo()
        return view

    def _apply_atom_colors(self, view: object, scheme: ColorScheme) -> None:
        for atom in self.molecule.structure.atoms:
            color = scheme.atom_color(atom.idx, self.molecule)
            radius = _SPHERE_RADIUS_HYDROGEN if atom.atomic_number == 1 else _SPHERE_RADIUS_HEAVY
            view.addStyle(
                {"serial": [atom.number]},
                {"sphere": {"color": color, "radius": radius},
                 "stick": {"color": color, "radius": _STICK_RADIUS}},
            )

    def _add_legend(self, view: object, scheme: ColorScheme) -> None:
        for i, (label, color) in enumerate(scheme.legend(self.molecule)):
            view.addLabel(
                label,
                {
                    "position": {"x": _LEGEND_X, "y": _LEGEND_Y_START + i * _LEGEND_Y_STEP, "z": 0},
                    "backgroundColor": color,
                    "fontColor": _legend_font_color(color),
                    "fontSize": _LEGEND_FONT_SIZE,
                    "backgroundOpacity": 0.85,
                    "useScreen": True,
                },
            )


def _legend_font_color(background_hex: str) -> str:
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(background_hex)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "#333333"


def _wrap_panels_with_toggles(
    scheme_htmls: list[str],
    names: list[str],
    width: int,
    height: int,
) -> str:
    encoded_panels = [
        base64.b64encode(html.encode("utf-8")).decode("ascii")
        for html in scheme_htmls
    ]
    button_html = "\n    ".join(
        f'<button class="toggle-btn{" active" if i == 0 else ""}" '
        f'onclick="showPanel({i}, this)">{name}</button>'
        for i, name in enumerate(names)
    )
    iframe_html = "\n    ".join(
        f'<iframe id="panel-{i}" class="viewer-panel{" active" if i == 0 else ""}" '
        f'src="data:text/html;base64,{encoded}" '
        f'width="{width}" height="{height}" frameborder="0"></iframe>'
        for i, encoded in enumerate(encoded_panels)
    )
    return f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 12px; background: #FAFAFA; }}
  h3 {{ margin: 0 0 10px 0; color: #333; }}
  #controls {{ margin-bottom: 10px; }}
  .toggle-btn {{
    padding: 7px 16px; margin: 0 4px 0 0; border: 1px solid #CCC;
    border-radius: 4px; background: #F5F5F5; cursor: pointer; font-size: 13px;
  }}
  .toggle-btn.active {{ background: #4C72B0; color: white; border-color: #4C72B0; }}
  .toggle-btn:hover:not(.active) {{ background: #E8E8E8; }}
  .viewer-panel {{ display: none; border: none; }}
  .viewer-panel.active {{ display: block; }}
</style>
</head>
<body>
<h3>Polymer 3D View</h3>
<div id="controls">
    {button_html}
</div>
<div id="viewer">
    {iframe_html}
</div>
<script>
  function showPanel(idx, btn) {{
    document.querySelectorAll('.viewer-panel').forEach((p, i) => {{
      p.classList.toggle('active', i === idx);
    }});
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }}
</script>
</body>
</html>"""
