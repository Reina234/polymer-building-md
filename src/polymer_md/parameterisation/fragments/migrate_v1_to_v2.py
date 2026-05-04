from __future__ import annotations

import json
from pathlib import Path


def migrate_v1_to_v2(path: Path) -> None:
    data = json.loads(path.read_text())
    if data.get("schema_version", "1") != "1":
        return
    data["records"] = [r for r in data["records"] if r.get("parameter_kind") != "dihedral"]
    data["schema_version"] = "2"
    path.write_text(json.dumps(data, indent=2))
