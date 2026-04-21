from __future__ import annotations

import logging
from dataclasses import dataclass

from openbabel import pybel
from rdkit import Chem

from polymer_md.geometry.base import ConformerGenerator
from polymer_md.geometry.exceptions import ConformerEmbeddingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OBabelConformerGenerator(ConformerGenerator):
    forcefield: str = "mmff94"
    steps: int = 500

    def embed(self, mol: Chem.Mol) -> Chem.Mol:
        smiles = Chem.MolToSmiles(mol)
        pybel_mol = pybel.readstring("smi", smiles)
        pybel_mol.make3D(forcefield=self.forcefield, steps=self.steps)
        sdf_string = pybel_mol.write("sdf")
        rdkit_mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        if rdkit_mol is None:
            raise ConformerEmbeddingError(
                "OBabel conformer generation produced an unparseable SDF. "
                "The molecule may be too complex for this force field."
            )
        logger.info(
            "OBabel conformer embedded with %s (%d steps).", self.forcefield, self.steps
        )
        return rdkit_mol
