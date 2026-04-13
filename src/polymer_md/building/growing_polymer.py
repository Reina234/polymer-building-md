from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Counter, List, Optional

from rdkit import Chem

from polymer_md.building.data_models import (
    AdditionPolymerResidue,
    MapLabels,
    PolymerisationLabels,
)
from polymer_md.core import ResidueInstance
from polymer_md.core.mol_atom import MolAtom
from polymer_md.utils import RDKitHelper


@dataclass(frozen=True)
class GrowingEnd:
    end_atom: MolAtom
    residue_id: str


class AdditionPolymer:

    def __init__(self):
        self._mol: Optional[Chem.rdchem.Mol] = None
        self._residue_instances: List[ResidueInstance] = []
        self._growing_end: Optional[GrowingEnd]
        self._instance_counts: Counter[str] = Counter()

    @property
    def mol(self) -> Chem.rdchem.Mol:
        if self._mol is None:
            raise ValueError(
                "Polymer not intialised yet, mol is None, please call .initalise()"
            )
        return self._mol

    @property
    def growing_end(self) -> GrowingEnd:
        if self._growing_end is None:
            raise ValueError(
                "No growing end present, polymer is either fully capped, or, hasn't been initialised()"
            )
        return self._growing_end

    def _get_random_site(self) -> PolymerisationLabels:
        return random.choice([MapLabels.HEAD, MapLabels.TAIL])

    def initialise(
        self,
        initial_residue: AdditionPolymerResidue,
        initial_site: Optional[PolymerisationLabels],
    ) -> None:
        residue = copy.deepcopy(initial_residue.mol)
        initial_site = initial_site or self._get_random_site()
        RDKitHelper.replace_with_placeholder(
            mol=residue,
            index_to_replace=initial_residue.get_polymerisation_atom(
                site=initial_site.other()
            ).idx,
            placeholder_map_num=MapLabels.CAP,
        )
        self._growing_end = GrowingEnd(
            end_atom=initial_residue.get_polymerisation_atom(site=initial_site),
            residue_id=initial_residue.id,
        )
        self._register_residue(
            residue=residue,
            atom_indices=self._get_new_residue_indices(residue_mol=initial_residue.mol),
        )

    def _get_new_residue_indices(self, residue_mol: Chem.Mol) -> list[int]:
        offset = self.mol.GetNumAtoms()
        return list(range(offset, offset + residue_mol.GetNumAtoms()))

    def _register_residue(
        self,
        residue: AdditionPolymerResidue,
        atom_indices: list[int],
    ) -> ResidueInstance:
        instance_number = self._instance_counts[residue.id]
        self._instance_counts[residue.id] += 1
        instance = ResidueInstance(
            residue_id=residue.id,
            instance_number=instance_number,
            atom_indices=frozenset(atom_indices),
        )
        self._residue_instances.append(instance)
        return instance
