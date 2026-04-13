from dataclasses import dataclass


@dataclass(frozen=True)
class Cap:
    smiles: str
    label: str


class BuiltinCap:
    HYDROGEN = Cap(smiles="*[H]", label="H")
    METHYL = Cap(smiles="*C", label="Me")
    HYDROXYL = Cap(smiles="*O", label="OH")
