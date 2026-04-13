from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Monomer:
    smiles: str
    label: Optional[str]
