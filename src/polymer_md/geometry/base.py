from __future__ import annotations

from abc import ABC, abstractmethod

from rdkit import Chem


class ConformerGenerator(ABC):
    @abstractmethod
    def embed(self, mol: Chem.Mol) -> Chem.Mol: ...
