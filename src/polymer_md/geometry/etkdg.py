from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.geometry.base import ConformerGenerator
from polymer_md.geometry.exceptions import ConformerEmbeddingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ETKDGConformerGenerator(ConformerGenerator):
    use_uff: bool = False
    random_seed: int = 42
    max_iterations: int = 1000

    def embed(self, mol: Chem.Mol) -> Chem.Mol:
        mol_with_hs = Chem.AddHs(mol)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = self.random_seed
        parameters.maxIterations = self.max_iterations
        result = AllChem.EmbedMolecule(mol_with_hs, parameters)
        if result == -1:
            raise ConformerEmbeddingError(
                f"ETKDG failed to embed the molecule after {self.max_iterations} iterations. "
                "Try increasing max_iterations or verify the molecule is chemically valid."
            )
        if self.use_uff:
            self._optimise_with_uff(mol_with_hs)
        logger.info(
            "ETKDG conformer embedded%s.",
            " and UFF-optimised" if self.use_uff else "",
        )
        return mol_with_hs

    @staticmethod
    def _optimise_with_uff(mol: Chem.Mol) -> None:
        result = AllChem.UFFOptimizeMolecule(mol)
        if result == -1:
            raise ConformerEmbeddingError(
                "UFF optimisation failed. The molecule may contain unsupported atom types."
            )
