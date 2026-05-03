from __future__ import annotations

from dataclasses import dataclass, field

import parmed as pmd
from rdkit import Chem

from polymer_md.conversion.gromacs_files import GromacsFiles


@dataclass(frozen=True)
class ParameterisedMolecule:
    structure: pmd.Structure
    mol: Chem.Mol
    source: GromacsFiles
    atom_metadata: dict[int, tuple[str, int]] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_gromacs_files(cls, files: GromacsFiles, mol: Chem.Mol) -> ParameterisedMolecule:
        structure = pmd.load_file(str(files.top), xyz=str(files.gro))
        return cls(structure=structure, mol=mol, source=files)
