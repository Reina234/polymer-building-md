from __future__ import annotations

import base64
import json
import re
import tempfile
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

from polymer_md.analysis.atom_deviation import AtomDeviation, compute_atom_deviations
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.visualisation._3d_base import is_jupyter, structure_to_pdb_string

_VIEWER_WIDTH = 820
_VIEWER_HEIGHT = 520
_SPHERE_RADIUS_HEAVY = 0.42
_STICK_RADIUS = 0.08
_CG_BEAD_RADIUS = 2.4
_CG_BOND_RADIUS = 0.45
_DEFAULT_MAX_PCT = 50.0
_N_COLORBAR_STEPS = 7
_UNCOVERED_COLOR = "#AAAAAA"

_DIFF_CMAP = "bwr"

_PARAM_LABELS = {
    "FORCE_CONSTANT": "Force constant",
    "EQUILIBRIUM_LENGTH": "Eq. length",
    "EQUILIBRIUM_ANGLE": "Eq. angle",
    "CHARGE": "Charge",
    "EPSILON": "LJ ε",
    "SIGMA": "LJ σ",
    "MASS": "Mass",
}


def _non_clashing_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _human_pattern(pattern: str) -> str:
    p = pattern
    p = re.sub(r"\[#6;A\]", "C", p)
    p = re.sub(r"\[#8;A\]", "O", p)
    p = re.sub(r"\[#7;A\]", "N", p)
    p = re.sub(r"\[#16;A\]", "S", p)
    p = re.sub(r"\[#6\]", "C", p)
    p = re.sub(r"\[#8\]", "O", p)
    p = re.sub(r"\[#7\]", "N", p)
    p = p.replace("-", "–").replace("~", "–")
    return p


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

    def save(self, output_path: Path) -> Path:
        output_path = _non_clashing_path(Path(output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self._build_html(), encoding="utf-8")
        return output_path

    def _show_jupyter(self) -> None:
        from IPython.display import HTML, display
        display(HTML(self._build_html()))

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
        instances = _group_monomer_instances(self.molecule.atom_metadata)

        atom_html = self._make_atom_viewer_html(deviation)
        cg_html = self._make_cg_viewer_html(deviation, instances)
        colorbar_html = _make_colorbar_html(self.max_pct)
        monomer_html = _make_monomer_sequence_html(instances, deviation.per_atom, self.max_pct)
        stats_html = _make_visual_stats_html(deviation.summaries, self.max_pct)

        return _assemble_html(
            atom_html, cg_html, colorbar_html, monomer_html, stats_html,
            self.width, self.height,
        )

    # ------------------------------------------------------------------
    # Atom-level viewer
    # ------------------------------------------------------------------

    def _make_atom_viewer_html(self, deviation: AtomDeviation) -> str:
        import py3Dmol

        pdb_str = structure_to_pdb_string(self.molecule.structure)
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(pdb_str, "pdb")
        view.setStyle({}, {"stick": {"color": "#CCCCCC", "radius": _STICK_RADIUS}})
        view.setStyle({"elem": "H"}, {"stick": {"color": "#E8E8E8", "radius": _STICK_RADIUS * 0.5}})
        self._apply_deviation_colors(view, deviation)
        self._add_monomer_labels(view)
        self._add_hover(view, deviation)
        view.setBackgroundColor("#F8F8F8")
        view.zoomTo()
        return _wrap_fragment(view._make_html())

    def _apply_deviation_colors(self, view: object, deviation: AtomDeviation) -> None:
        cmap_obj = matplotlib.colormaps[_DIFF_CMAP]
        norm = mcolors.TwoSlopeNorm(vmin=-self.max_pct, vcenter=0.0, vmax=self.max_pct)

        for atom in self.molecule.structure.atoms:
            if atom.atomic_number == 1:
                continue
            if atom.idx not in deviation.per_atom:
                view.addStyle(
                    {"serial": [atom.number]},
                    {"sphere": {"color": _UNCOVERED_COLOR, "radius": _SPHERE_RADIUS_HEAVY * 0.6}},
                )
                continue
            pct = deviation.per_atom[atom.idx]
            clamped = max(-self.max_pct, min(self.max_pct, pct))
            color = mcolors.to_hex(cmap_obj(norm(clamped)))
            view.addStyle(
                {"serial": [atom.number]},
                {
                    "sphere": {"color": color, "radius": _SPHERE_RADIUS_HEAVY},
                    "stick": {"color": color, "radius": _STICK_RADIUS},
                },
            )

    def _add_monomer_labels(self, view: object) -> None:
        if not self.molecule.atom_metadata:
            return
        idx_to_atom = {a.idx: a for a in self.molecule.structure.atoms}
        instances = _group_monomer_instances(self.molecule.atom_metadata)
        for residue_id, atom_indices in instances:
            coords = [
                (idx_to_atom[i].xx, idx_to_atom[i].xy, idx_to_atom[i].xz)
                for i in atom_indices if i in idx_to_atom
            ]
            if not coords:
                continue
            view.addLabel(
                residue_id,
                {
                    "position": {
                        "x": float(np.mean([c[0] for c in coords])),
                        "y": float(np.mean([c[1] for c in coords])),
                        "z": float(np.mean([c[2] for c in coords])),
                    },
                    "fontSize": 9,
                    "fontColor": "#111111",
                    "backgroundColor": "#FFFFFF",
                    "backgroundOpacity": 0.75,
                    "borderThickness": 0,
                    "inFront": True,
                },
            )

    def _add_hover(self, view: object, deviation: AtomDeviation) -> None:
        atom_js_data = _build_atom_js_data(self.molecule, deviation)
        view.setHoverable(
            {},
            True,
            f"""function(atom, viewer, event, container) {{
                if (atom._label) return;
                var data = {atom_js_data};
                var info = data[atom.serial];
                if (!info) return;
                var text = atom.atom + " (" + info.t + ")";
                if (info.p.length > 0) {{
                    text += " | avg dev: " + info.avg + "%";
                    for (var i = 0; i < info.p.length; i++) {{
                        var p = info.p[i];
                        text += " | " + p.n + ": " + p.v.toFixed(3) + " vs " + p.r.toFixed(3) + " (" + (p.d >= 0 ? "+" : "") + p.d.toFixed(1) + "%)";
                    }}
                }} else {{
                    text += " | not covered by any fragment";
                }}
                atom._label = viewer.addLabel(text, {{
                    position: atom, backgroundColor: "#1a1a2e", fontColor: "#FFFFFF",
                    fontSize: 11, backgroundOpacity: 0.92, borderThickness: 0,
                    inFront: true
                }});
            }}""",
            """function(atom, viewer) {
                if (atom._label) { viewer.removeLabel(atom._label); delete atom._label; }
            }""",
        )

    # ------------------------------------------------------------------
    # CG bead viewer
    # ------------------------------------------------------------------

    def _make_cg_viewer_html(
        self,
        deviation: AtomDeviation,
        instances: list[tuple[str, list[int]]],
    ) -> str:
        import py3Dmol

        view = py3Dmol.view(width=self.width, height=self.height)

        cmap_obj = matplotlib.colormaps[_DIFF_CMAP]
        norm = mcolors.TwoSlopeNorm(vmin=-self.max_pct, vcenter=0.0, vmax=self.max_pct)
        idx_to_atom = {a.idx: a for a in self.molecule.structure.atoms}

        centroids: list[tuple[float, float, float] | None] = []
        for residue_id, atom_indices in instances:
            coords = [
                (idx_to_atom[i].xx, idx_to_atom[i].xy, idx_to_atom[i].xz)
                for i in atom_indices if i in idx_to_atom
            ]
            if not coords:
                centroids.append(None)
                continue

            cx = float(np.mean([c[0] for c in coords]))
            cy = float(np.mean([c[1] for c in coords]))
            cz = float(np.mean([c[2] for c in coords]))
            centroids.append((cx, cy, cz))

            devs = [deviation.per_atom[i] for i in atom_indices if i in deviation.per_atom]
            covered = len(devs)
            total = len(atom_indices)
            if devs:
                avg = float(np.mean(devs))
                clamped = max(-self.max_pct, min(self.max_pct, avg))
                color = mcolors.to_hex(cmap_obj(norm(clamped)))
                label_suffix = f" {avg:+.1f}%"
            else:
                color = _UNCOVERED_COLOR
                label_suffix = " (no data)"

            view.addSphere({
                "center": {"x": cx, "y": cy, "z": cz},
                "radius": _CG_BEAD_RADIUS,
                "color": color,
                "opacity": 0.88,
            })
            view.addLabel(
                f"{residue_id}{label_suffix}",
                {
                    "position": {"x": cx, "y": cy + _CG_BEAD_RADIUS + 1.2, "z": cz},
                    "fontSize": 11,
                    "fontColor": "#111111",
                    "backgroundColor": "#FFFFFF",
                    "backgroundOpacity": 0.82,
                    "borderThickness": 0,
                    "inFront": True,
                },
            )

        for i in range(len(centroids) - 1):
            c1, c2 = centroids[i], centroids[i + 1]
            if c1 is None or c2 is None:
                continue
            view.addCylinder({
                "start": {"x": c1[0], "y": c1[1], "z": c1[2]},
                "end": {"x": c2[0], "y": c2[1], "z": c2[2]},
                "radius": _CG_BOND_RADIUS,
                "color": "#888888",
                "opacity": 0.5,
                "dashed": False,
            })

        view.setBackgroundColor("#F8F8F8")
        view.zoomTo()
        return _wrap_fragment(view._make_html())


# ------------------------------------------------------------------
# Helper builders
# ------------------------------------------------------------------

def _wrap_fragment(fragment: str) -> str:
    return (
        "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\"></head>\n"
        "<body style=\"margin:0;padding:0;background:#F8F8F8;\">\n"
        f"{fragment}\n</body>\n</html>"
    )


def _build_atom_js_data(molecule: ParameterisedMolecule, deviation: AtomDeviation) -> str:
    data: dict[int, dict] = {}
    for atom in molecule.structure.atoms:
        details = deviation.per_atom_details.get(atom.idx, [])
        params = [
            {
                "n": label.split(" / ")[-1],
                "v": round(our_val, 4),
                "r": round(ref_mean, 4),
                "d": round((our_val - ref_mean) / (abs(ref_mean) + 1e-10) * 100, 1),
            }
            for label, our_val, ref_mean in details
        ]
        avg_dev = round(float(np.mean([p["d"] for p in params])), 1) if params else 0.0
        data[atom.number] = {"t": atom.type or atom.name, "p": params, "avg": avg_dev}
    return json.dumps(data, separators=(",", ":"))


def _group_monomer_instances(
    atom_metadata: dict[int, tuple[str, int]],
) -> list[tuple[str, list[int]]]:
    instances: list[tuple[str, list[int]]] = []
    current_id: str | None = None
    current_atoms: list[int] = []
    for idx in sorted(atom_metadata.keys()):
        residue_id, position = atom_metadata[idx]
        if position == 0 and current_atoms:
            instances.append((current_id, current_atoms))  # type: ignore[arg-type]
            current_atoms = []
        current_id = residue_id
        current_atoms.append(idx)
    if current_atoms:
        instances.append((current_id, current_atoms))  # type: ignore[arg-type]
    return instances


# ------------------------------------------------------------------
# HTML sections
# ------------------------------------------------------------------

def _make_colorbar_html(max_pct: float) -> str:
    cmap_obj = matplotlib.colormaps[_DIFF_CMAP]
    norm = mcolors.TwoSlopeNorm(vmin=-max_pct, vcenter=0.0, vmax=max_pct)
    n = 120
    stops = ", ".join(
        f"{mcolors.to_hex(cmap_obj(norm(v)))} {i * 100 / (n - 1):.1f}%"
        for i, v in enumerate(np.linspace(-max_pct, max_pct, n))
    )
    ticks = np.linspace(-max_pct, max_pct, _N_COLORBAR_STEPS)
    tick_html = "\n".join(
        f'<div style="position:absolute;left:{(v + max_pct) / (2 * max_pct) * 100:.1f}%;'
        f'transform:translateX(-50%);top:22px;font-size:11px;color:#555">'
        f'{v:+.0f}%</div>'
        for v in ticks
    )
    legend_items = (
        f'<span style="margin-right:14px"><span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{_UNCOVERED_COLOR};vertical-align:middle;margin-right:4px"></span>'
        f'not covered</span>'
    )
    return f"""\
<div style="margin:8px 0 12px 0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 14px;">
  <div style="font-size:12px;font-weight:600;color:#333;margin-bottom:6px;">
    % deviation from reference &nbsp; {legend_items}
  </div>
  <div style="position:relative;height:44px;">
    <div style="height:20px;border-radius:4px;background:linear-gradient(to right,{stops});box-shadow:0 1px 3px rgba(0,0,0,.1)"></div>
    {tick_html}
  </div>
</div>"""


def _make_monomer_sequence_html(
    instances: list[tuple[str, list[int]]],
    per_atom: dict[int, float],
    max_pct: float,
) -> str:
    if not instances:
        return ""

    cmap_obj = matplotlib.colormaps[_DIFF_CMAP]
    norm = mcolors.TwoSlopeNorm(vmin=-max_pct, vcenter=0.0, vmax=max_pct)

    per_type: dict[str, dict] = {}
    blocks = []
    for residue_id, atom_indices in instances:
        n_total = len(atom_indices)
        devs = [per_atom[idx] for idx in atom_indices if idx in per_atom]
        n_covered = len(devs)
        coverage_pct = n_covered / n_total * 100 if n_total else 0.0

        if devs:
            avg = float(np.mean(devs))
            clamped = max(-max_pct, min(max_pct, avg))
            color = mcolors.to_hex(cmap_obj(norm(clamped)))
            tooltip = f"{residue_id}: avg {avg:+.1f}% | coverage {n_covered}/{n_total} ({coverage_pct:.0f}%)"
        else:
            color = _UNCOVERED_COLOR
            avg = 0.0
            tooltip = f"{residue_id}: no coverage ({n_total} atoms)"

        luma = sum(int(color.lstrip("#")[i:i+2], 16) * w
                   for i, w in zip((0, 2, 4), (0.299, 0.587, 0.114)))
        text_color = "#fff" if luma < 140 else "#111"

        blocks.append(
            f'<div title="{tooltip}" style="display:inline-block;width:30px;height:30px;'
            f'background:{color};border:1px solid rgba(0,0,0,.15);border-radius:4px;margin:1px;'
            f'text-align:center;line-height:30px;font-size:8.5px;font-weight:bold;'
            f'color:{text_color};overflow:hidden;cursor:default;">{residue_id[:3]}</div>'
        )

        td = per_type.setdefault(residue_id, {"devs": [], "n_total": 0, "n_covered": 0})
        td["devs"].extend(devs)
        td["n_total"] += n_total
        td["n_covered"] += n_covered

    type_rows = []
    for residue_id, td in sorted(per_type.items()):
        devs = td["devs"]
        n_total = td["n_total"]
        n_covered = td["n_covered"]
        coverage_pct = n_covered / n_total * 100 if n_total else 0.0
        mean = float(np.mean(devs)) if devs else 0.0
        std = float(np.std(devs)) if devs else 0.0
        diff_color = "#cc3333" if mean > 0 else "#2255bb"
        bar_w = min(abs(mean) / max_pct * 100, 100)
        bar_color = "#cc3333" if mean > 0 else "#2255bb"
        type_rows.append(
            f"<tr>"
            f"<td style='padding:5px 10px;font-weight:bold'>{residue_id}</td>"
            f"<td style='padding:5px 10px'>{n_covered}/{n_total} ({coverage_pct:.0f}%)</td>"
            f"<td style='padding:5px 10px;color:{diff_color};font-weight:bold'>{mean:+.1f}%</td>"
            f"<td style='padding:5px 10px'>{std:.1f}%</td>"
            f"<td style='padding:5px 10px'>"
            f"<div style='display:inline-block;width:{bar_w:.0f}px;height:10px;background:{bar_color};border-radius:2px;min-width:2px'></div>"
            f"</td>"
            f"</tr>"
        )

    return f"""\
<h4 style="margin:16px 0 5px 0;color:#333">Chain sequence <span style="font-weight:normal;font-size:12px;color:#777">(hover each block for detail)</span></h4>
<div style="overflow-x:auto;padding:4px 0;white-space:nowrap;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px 8px;">{" ".join(blocks)}</div>
<h4 style="margin:14px 0 5px 0;color:#333">Per-monomer-type summary</h4>
<table style="font-size:13px;border-collapse:collapse;width:auto;">
  <thead>
    <tr style="background:#4C72B0;color:white">
      <th style="padding:6px 10px;text-align:left">Monomer</th>
      <th style="padding:6px 10px;text-align:left">Coverage</th>
      <th style="padding:6px 10px;text-align:left">Mean deviation</th>
      <th style="padding:6px 10px;text-align:left">Std</th>
      <th style="padding:6px 10px;text-align:left">Bar</th>
    </tr>
  </thead>
  <tbody>{"".join(type_rows)}</tbody>
</table>"""


def _make_visual_stats_html(summaries, max_pct: float) -> str:
    if not summaries:
        return "<p style='color:#888;margin-top:12px'>No fragment statistics available.</p>"

    rows = sorted(summaries, key=lambda s: abs(s.pct_diff), reverse=True)
    row_html = ""
    for s in rows:
        bar_w = min(abs(s.pct_diff) / max_pct * 120, 120)
        bar_color = "#cc3333" if s.pct_diff > 0 else "#2255bb"
        diff_color = bar_color
        param_label = _PARAM_LABELS.get(s.parameter_name, s.parameter_name)
        pattern_label = _human_pattern(s.pattern)
        row_html += (
            f"<tr>"
            f"<td style='padding:5px 12px'><b>{pattern_label}</b></td>"
            f"<td style='padding:5px 12px;color:#555'>{param_label}</td>"
            f"<td style='padding:5px 12px;font-size:12px;color:#444'>"
            f"ours {s.our_mean:.3f} <span style='color:#999'>n={s.our_count}</span>"
            f"<br>ref&nbsp;&nbsp; {s.ref_mean:.3f} <span style='color:#999'>n={s.ref_count}</span>"
            f"</td>"
            f"<td style='padding:5px 12px'>"
            f"<div style='display:flex;align-items:center;gap:6px'>"
            f"<div style='width:{bar_w:.0f}px;height:12px;background:{bar_color};border-radius:3px;min-width:2px'></div>"
            f"<span style='color:{diff_color};font-weight:bold;font-size:13px'>{s.pct_diff:+.1f}%</span>"
            f"</div></td>"
            f"</tr>\n"
        )
    return f"""\
<table style="font-size:13px;border-collapse:collapse;width:100%;">
  <thead>
    <tr style="background:#4C72B0;color:white;">
      <th style="padding:7px 12px;text-align:left">Bond/atom</th>
      <th style="padding:7px 12px;text-align:left">Parameter</th>
      <th style="padding:7px 12px;text-align:left">Values</th>
      <th style="padding:7px 12px;text-align:left">Deviation</th>
    </tr>
  </thead>
  <tbody>
{row_html}  </tbody>
</table>"""


def _assemble_html(
    atom_viewer_html: str,
    cg_viewer_html: str,
    colorbar_html: str,
    monomer_html: str,
    stats_html: str,
    width: int,
    height: int,
) -> str:
    atom_enc = base64.b64encode(atom_viewer_html.encode("utf-8")).decode("ascii")
    cg_enc = base64.b64encode(cg_viewer_html.encode("utf-8")).decode("ascii")
    return f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 14px; background: #F5F5F5; color: #222; }}
  h3 {{ margin: 0 0 10px 0; color: #222; font-size: 17px; }}
  h4 {{ color: #333; font-size: 14px; }}
  iframe {{ display: block; border: none; border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
  .btn {{ padding: 6px 18px; border: 1px solid #4C72B0; border-radius: 4px;
          background: #fff; color: #4C72B0; font-size: 13px; cursor: pointer;
          font-family: inherit; transition: background .15s; }}
  .btn.active {{ background: #4C72B0; color: #fff; }}
  .btn:hover:not(.active) {{ background: #eef2ff; }}
  table tr:nth-child(even) {{ background: #F5F7FA; }}
  table td, table th {{ border-bottom: 1px solid #E0E0E0; }}
</style>
</head>
<body>
<h3>Parameter Deviation from Reference</h3>
{colorbar_html}
<div style="margin-bottom:8px;display:flex;gap:6px;align-items:center">
  <button class="btn active" id="btn-atom" onclick="switchView('atom')">Atom view</button>
  <button class="btn" id="btn-cg" onclick="switchView('cg')">CG view</button>
  <span style="font-size:12px;color:#777;margin-left:6px">
    Atom: hover atoms for detail &nbsp;|&nbsp; CG: one bead per monomer, coloured by avg deviation
  </span>
</div>
<iframe id="view-atom" src="data:text/html;base64,{atom_enc}" width="{width}" height="{height}" frameborder="0"></iframe>
<iframe id="view-cg"  src="data:text/html;base64,{cg_enc}"  width="{width}" height="{height}" frameborder="0" style="display:none"></iframe>
{monomer_html}
<h4 style="margin:18px 0 6px 0">Fragment parameter comparison</h4>
{stats_html}
<script>
function switchView(type) {{
    document.getElementById('view-atom').style.display = type === 'atom' ? 'block' : 'none';
    document.getElementById('view-cg').style.display  = type === 'cg'   ? 'block' : 'none';
    document.getElementById('btn-atom').className = 'btn' + (type === 'atom' ? ' active' : '');
    document.getElementById('btn-cg').className   = 'btn' + (type === 'cg'   ? ' active' : '');
}}
</script>
</body>
</html>"""
