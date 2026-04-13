from dataclasses import dataclass


@dataclass(frozen=True)
class Cap:
    smiles: str
    label: str
    id: str


class BuiltinCap:
    HYDROGEN = Cap(smiles="*[H]", label="H", id="H")
    METHYL = Cap(smiles="*C", label="Me", id="Me")
    HYDROXYL = Cap(smiles="*O", label="OH", id="OH")
