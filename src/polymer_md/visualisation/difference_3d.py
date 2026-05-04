from __future__ import annotations

import json
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

from polymer_md.analysis.atom_deviation import AtomDeviation, compute_atom_deviations
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.visualisation._3d_base import is_jupyter, structure_to_pdb_string
from polymer_md.visualisation.polymer_3d import _legend_font_color

_VIEWER_WIDTH = 820
_VIEWER_HEIGHT = 500
_SPHERE_RADIUS_HEAVY = 0.28
_SPHERE_RADIUS_HYDROGEN = 0.14
_STICK_RADIUS = 0.08
_DEFAULT_MAX_PCT = 50.0
_N_COLORBAR_STEPS = 7

_DIFF_CMAP = "bwr"


@dataclass
class DifferenceViewer:
    molecule: ParameterisedMolecule
    reference: ParameterisedMolecule
    fragments: list[Fragment]
    width: int = _VIEWER_WIDTH
    height: int = _VIEWER_HEIGHT
    max_pct: float = _DEFAULT_MAX_PCT

    def show(self, output_path: Path | None = None) -> None:
        if is_jupyter():
            self._show_jupyter()
        else:
            self._show_standalone(output_path)

    def save(self, output_path: Path) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(self._build_html(), encoding="utf-8")

    def _show_jupyter(self) -> None:
        from IPython.display import HTML, display
        html = self._build_html()
        display(HTML(html))

    def _show_standalone(self, output_path: Path | None = None) -> None:
        html = self._build_html()
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

    def _build_html(self) -> str:
        deviation = compute_atom_deviations(self.molecule, self.reference, self.fragments)
        viewer_html = self._make_viewer_html(deviation)
        colorbar_html = _make_colorbar_html(self.max_pct)
        table_html = _make_stats_table_html(deviation.summaries)
        return _assemble_difference_html(
            viewer_html, colorbar_html, table_html, self.width, self.height
        )

    def _make_viewer_html(self, deviation: AtomDeviation) -> str:
        import py3Dmol

        pdb_str = structure_to_pdb_string(self.molecule.structure)
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(pdb_str, "pdb")
        view.setStyle({}, {"stick": {"colorscheme": "greyCarbon", "radius": _STICK_RADIUS}})
        self._apply_deviation_colors(view, deviation)
        self._add_hover(view, deviation)
        view.setBackgroundColor("#FAFAFA")
        view.zoomTo()
        fragment = view._make_html()
        return (
            "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"></head>\n"
            "<body style=\"margin:0;padding:0;background:#FAFAFA;\">\n"
            f"{fragment}\n</body>\n</html>"
        )

    def _apply_deviation_colors(self, view: object, deviation: AtomDeviation) -> None:
        cmap_obj = matplotlib.colormaps[_DIFF_CMAP]
        norm = mcolors.TwoSlopeNorm(vmin=-self.max_pct, vcenter=0.0, vmax=self.max_pct)

        for atom in self.molecule.structure.atoms:
            pct = deviation.per_atom.get(atom.idx, 0.0)
            clamped = max(-self.max_pct, min(self.max_pct, pct))
            color = mcolors.to_hex(cmap_obj(norm(clamped)))
            radius = _SPHERE_RADIUS_HYDROGEN if atom.atomic_number == 1 else _SPHERE_RADIUS_HEAVY
            view.addStyle(
                {"serial": [atom.number]},
                {
                    "sphere": {"color": color, "radius": radius},
                    "stick": {"color": color, "radius": _STICK_RADIUS},
                },
            )

    def _add_hover(self, view: object, deviation: AtomDeviation) -> None:
        atom_js_data = _build_atom_js_data(self.molecule, deviation)
        view.setHoverable(
            {},
            True,
            f"""function(atom, viewer, event, container) {{
                var data = {atom_js_data};
                var info = data[atom.serial];
                if (!atom.label && info) {{
                    var lines = [atom.atom + " (" + info.t + ")"];
                    for (var i = 0; i < info.p.length; i++) {{
                        var p = info.p[i];
                        lines.push(p.n + ": " + p.v.toFixed(4) + " (ref " + p.r.toFixed(4) + ", " + (p.d > 0 ? "+" : "") + p.d.toFixed(1) + "%)");
                    }}
                    atom.label = viewer.addLabel(lines.join("\\n"), {{
                        position: atom, backgroundColor: "#222222", fontColor: "#FFFFFF",
                        fontSize: 11, backgroundOpacity: 0.9, borderThickness: 0
                    }});
                }}
            }}""",
            """function(atom, viewer) {
                if (atom.label) { viewer.removeLabel(atom.label); delete atom.label; }
            }""",
        )


def _build_atom_js_data(molecule: ParameterisedMolecule, deviation: AtomDeviation) -> str:
    data: dict[int, dict] = {}
    for atom in molecule.structure.atoms:
        details = deviation.per_atom_details.get(atom.idx, [])
        params = [
            {"n": label, "v": round(our_val, 6), "r": round(ref_mean, 6),
             "d": round((our_val - ref_mean) / (abs(ref_mean) + 1e-10) * 100, 1)}
            for label, our_val, ref_mean in details
        ]
        data[atom.number] = {"t": atom.type or atom.name, "p": params}
    return json.dumps(data)


def _make_colorbar_html(max_pct: float) -> str:
    cmap_obj = matplotlib.colormaps[_DIFF_CMAP]
    norm = mcolors.TwoSlopeNorm(vmin=-max_pct, vcenter=0.0, vmax=max_pct)
    n = 100
    stops = " ".join(
        f"{mcolors.to_hex(cmap_obj(norm(v)))} {i * 100 // (n - 1)}%"
        for i, v in enumerate(np.linspace(-max_pct, max_pct, n))
    )
    step = 2 * max_pct / (_N_COLORBAR_STEPS - 1)
    ticks = np.linspace(-max_pct, max_pct, _N_COLORBAR_STEPS)
    tick_html = "\n".join(
        f'<div style="position:absolute;left:{(v + max_pct) / (2 * max_pct) * 100:.1f}%;'
        f'transform:translateX(-50%);top:22px;font-size:11px;color:#555">'
        f'{v:+.0f}%</div>'
        for v in ticks
    )
    return f"""\
<div style="margin:10px 0 8px 0;">
  <div style="font-size:12px;font-weight:bold;color:#333;margin-bottom:4px;">
    % deviation from reference (blue = lower, red = higher)
  </div>
  <div style="position:relative;height:40px;">
    <div style="height:18px;border-radius:3px;background:linear-gradient(to right,{stops});"></div>
    {tick_html}
  </div>
</div>"""


def _make_stats_table_html(summaries) -> str:
    if not summaries:
        return "<p style='color:#888'>No fragment statistics available.</p>"

    rows = sorted(summaries, key=lambda s: abs(s.pct_diff), reverse=True)
    row_html = ""
    for s in rows:
        diff_color = "#cc3333" if s.pct_diff > 0 else "#2255bb"
        row_html += (
            f"<tr>"
            f"<td><code>{s.pattern}</code></td>"
            f"<td>{s.parameter_name}</td>"
            f"<td>{s.our_mean:.4f} ± {s.our_std:.4f} (n={s.our_count})</td>"
            f"<td>{s.ref_mean:.4f} ± {s.ref_std:.4f} (n={s.ref_count})</td>"
            f"<td style='color:{diff_color};font-weight:bold'>{s.pct_diff:+.1f}%</td>"
            f"</tr>\n"
        )
    return f"""\
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:6px;">
  <thead>
    <tr style="background:#4C72B0;color:white;">
      <th style="padding:7px 10px;text-align:left">Pattern</th>
      <th style="padding:7px 10px;text-align:left">Parameter</th>
      <th style="padding:7px 10px;text-align:left">Ours (mean ± std)</th>
      <th style="padding:7px 10px;text-align:left">Reference (mean ± std)</th>
      <th style="padding:7px 10px;text-align:left">% diff</th>
    </tr>
  </thead>
  <tbody>
{row_html}  </tbody>
</table>"""


def _assemble_difference_html(
    viewer_html: str,
    colorbar_html: str,
    table_html: str,
    width: int,
    height: int,
) -> str:
    import base64
    encoded = base64.b64encode(viewer_html.encode("utf-8")).decode("ascii")
    return f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 12px; background: #FAFAFA; }}
  h3 {{ margin: 0 0 8px 0; color: #333; }}
  iframe {{ display: block; border: none; }}
  table tr:nth-child(even) {{ background: #F5F7FA; }}
  table td, table th {{ border-bottom: 1px solid #E0E0E0; }}
  code {{ font-size: 12px; }}
</style>
</head>
<body>
<h3>Parameter Deviation from Reference</h3>
{colorbar_html}
<iframe src="data:text/html;base64,{encoded}" width="{width}" height="{height}" frameborder="0"></iframe>
<h4 style="margin:14px 0 6px 0;color:#333">Fragment Statistics</h4>
{table_html}
</body>
</html>"""
