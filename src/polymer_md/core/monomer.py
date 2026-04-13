from rdkit import Chem
from pydantic import BaseModel, ConfigDict, field_validator


class Monomer(BaseModel):
    model_config = ConfigDict(frozen=True)

    smiles: str
    name: str | None = None

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, value: str) -> str:
        if Chem.rdmolfiles.MolFromSmiles(value) is None:
            raise ValueError(f"Invalid SMILES: {value!r}")
        return value
