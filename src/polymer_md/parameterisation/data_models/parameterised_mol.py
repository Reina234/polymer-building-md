from __future__ import annotations

from dataclasses import dataclass

import parmed as pmd
from rdkit import Chem

from polymer_md.conversion.gromacs_files import GromacsFiles


@dataclass(frozen=True)
class ParameterisedMolecule:
    structure: pmd.Structure
    mol: Chem.Mol
    source: GromacsFiles

    @classmethod
    def from_gromacs_files(cls, files: GromacsFiles, mol: Chem.Mol) -> ParameterisedMolecule:
        structure = pmd.load_file(str(files.top), xyz=str(files.gro))
        return cls(structure=structure, mol=mol, source=files)
