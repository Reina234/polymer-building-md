from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeInput:
    smiles: str
    label: str | None = None
