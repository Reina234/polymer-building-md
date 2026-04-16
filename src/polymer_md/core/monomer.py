from __future__ import annotations

from dataclasses import dataclass

from polymer_md.core.molecule_input import MoleculeInput


@dataclass(frozen=True)
class Monomer(MoleculeInput):
    pass
