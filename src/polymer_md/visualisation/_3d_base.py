from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

import parmed as pmd


def is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and hasattr(shell, "kernel")
    except ImportError:
        return False


def structure_to_pdb_string(structure: pmd.Structure) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        path = f.name
    structure.save(path, overwrite=True)
    return Path(path).read_text()


def display_3d(view: object) -> object:
    if is_jupyter():
        view.show()  # type: ignore[attr-defined]
        return view
    html = view._make_html()  # type: ignore[attr-defined]
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html)
        path = f.name
    webbrowser.open(f"file://{path}")
    return view
