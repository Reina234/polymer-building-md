from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from rdkit import Chem

from polymer_md.building.data_models import (
    AdditionPolymerResidue,
    MapLabels,
    PolymerisationLabels,
)
from polymer_md.core import Cap, Polymer, ResidueInstance
from polymer_md.core.mol_atom import MolAtom
from polymer_md.utils import RDKitHelper


@dataclass(frozen=True)
class AdditionPolymerEnd:
    map_num: MapLabels
    residue_id: str


class AdditionPolymer:
    def __init__(self) -> None:
        self._mol: Optional[Chem.rdchem.Mol] = None
        self._growing_end: Optional[AdditionPolymerEnd] = None
        self._inactive_end: Optional[AdditionPolymerEnd] = None
        self._residue_instances: List[ResidueInstance] = []
        self._instance_counts: Counter[str] = Counter()

    @property
    def mol(self) -> Chem.rdchem.Mol:
        if self._mol is None:
            raise ValueError("Polymer not initialised, call .initialise() first.")
        return self._mol

    @property
    def growing_end(self) -> AdditionPolymerEnd:
        if self._growing_end is None:
            raise ValueError(
                "No growing end — polymer is either uninitialised or fully capped."
            )
        return self._growing_end

    @property
    def growing_end_atom(self) -> MolAtom:
        return MolAtom(
            mol=self.mol,
            idx=RDKitHelper.get_site_idx(
                mol=self.mol,
                atom_num=0,
                map_num=self.growing_end.map_num,
            ),
        )

    @property
    def inactive_end_atom(self) -> MolAtom:
        if self._inactive_end is None:
            raise ValueError("No inactive end — polymer has not been initialised.")
        return MolAtom(
            mol=self.mol,
            idx=RDKitHelper.get_site_idx(
                mol=self.mol,
                atom_num=0,
                map_num=MapLabels.INACTIVE_END,
            ),
        )

    def initialise(
        self,
        residue: AdditionPolymerResidue,
        site: Optional[PolymerisationLabels] = None,
    ) -> None:
        site = site or self._random_site()

        mol = RDKitHelper.replace_with_placeholder(
            mol=residue.mol,
            index_to_replace=residue.get_polymerisation_atom(site.other()).idx,
            placeholder_map_num=MapLabels.INACTIVE_END,
        )

        self._mol = mol
        self._growing_end = AdditionPolymerEnd(map_num=site, residue_id=residue.id)
        self._inactive_end = AdditionPolymerEnd(
            map_num=MapLabels.INACTIVE_END, residue_id=residue.id
        )
        self._register_residue(
            residue=residue,
            atom_indices=list(range(mol.GetNumAtoms())),
        )

    def add(
        self,
        residue: AdditionPolymerResidue,
        site: Optional[PolymerisationLabels] = None,
    ) -> None:
        site = site or self._random_site()
        new_indices = self._next_residue_indices(residue.mol)

        self._mol = RDKitHelper.single_bond_join_at_wildcard_sites(
            site1=self.growing_end_atom,
            site2=residue.get_polymerisation_atom(site),
        )
        self._growing_end = AdditionPolymerEnd(
            map_num=site.other(), residue_id=residue.id
        )
        self._register_residue(residue=residue, atom_indices=new_indices)

    def export(self, cap: Cap) -> Polymer:
        mol = self._apply_cap(self.growing_end_atom, cap)

        inactive_end_atom = MolAtom(
            mol=mol,
            idx=RDKitHelper.get_site_idx(
                mol=mol,
                atom_num=0,
                map_num=MapLabels.INACTIVE_END,
            ),
        )
        mol = self._apply_cap(inactive_end_atom, cap)

        return Polymer(
            mol=mol,
            residue_instances=list(self._residue_instances),
        )

    def _apply_cap(self, end_atom: MolAtom, cap: Cap) -> Chem.Mol:
        cap_mol = Chem.rdmolfiles.MolFromSmiles(cap.smiles)
        cap_site = MolAtom(
            mol=cap_mol,
            idx=RDKitHelper.get_site_idx(mol=cap_mol, atom_num=0, map_num=0),
        )
        return RDKitHelper.single_bond_join_at_wildcard_sites(
            site1=end_atom,
            site2=cap_site,
        )

    def _random_site(self) -> PolymerisationLabels:
        return random.choice([MapLabels.HEAD, MapLabels.TAIL])

    def _next_residue_indices(self, residue_mol: Chem.Mol) -> list[int]:
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
