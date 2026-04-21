from __future__ import annotations

from dataclasses import dataclass, field

import parmed as pmd
from rdkit import Chem

from polymer_md.building.data_models.trimer import TrimerResult
from polymer_md.conversion.gromacs_files import GromacsFiles


@dataclass(frozen=True)
class ParameterisedTrimer:
    trimer_result: TrimerResult
    gromacs_files: GromacsFiles
    structure: pmd.Structure = field(compare=False, hash=False)
    mol_3d: Chem.Mol = field(compare=False, hash=False)
