from dataclasses import dataclass
from enum import StrEnum

from polymer_md.utils import FileHelper, PathType


class FileFormats(StrEnum):
    PDB = "pdb"
    MOL2 = "mol2"
    GRO = "gro"
    ITP = "itp"
    TOP = "top"
    SDF = "sdf"


@dataclass(frozen=True)
class PathInput:
    path: PathType

    @property
    def format(self) -> str:
        return FileHelper.get_suffix_type(path=self.path)
