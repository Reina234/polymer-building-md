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
from polymer_md.core import RESIDUE_TAG, Cap, Polymer, ResidueInstance, ResidueType
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
        self._tag_counter: int = 0

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
        mol = self._prepare_initial_residue(residue, site)
        tags, mol = self._tag_atoms(mol)
        self._mol = mol
        self._set_initial_ends(residue=residue, site=site)
        self._register_residue(
            residue=residue, tags=tags, residue_type=ResidueType.MONOMER
        )

    def add(
        self,
        residue: AdditionPolymerResidue,
        site: Optional[PolymerisationLabels] = None,
    ) -> None:
        site = site or self._random_site()
        tags, tagged_mol = self._tag_atoms(residue.mol)
        self._mol = self._join_incoming(tagged_mol=tagged_mol, site=site)
        self._growing_end = AdditionPolymerEnd(
            map_num=site.other(), residue_id=residue.id
        )
        self._register_residue(
            residue=residue, tags=tags, residue_type=ResidueType.MONOMER
        )

    def export(self, cap: Cap) -> Polymer:
        cap_mol_growing, cap_mol_inactive = self._prepare_cap_mols(cap)
        growing_tags, cap_mol_growing = self._tag_atoms(cap_mol_growing)
        inactive_tags, cap_mol_inactive = self._tag_atoms(cap_mol_inactive)
        mol = self._join_caps(
            cap_mol_growing=cap_mol_growing, cap_mol_inactive=cap_mol_inactive
        )
        cap_instances = self._make_cap_instances(
            cap=cap, growing_tags=growing_tags, inactive_tags=inactive_tags
        )
        return Polymer(
            mol=mol,
            residue_instances=list(self._residue_instances) + cap_instances,
        )

    def _prepare_initial_residue(
        self,
        residue: AdditionPolymerResidue,
        site: PolymerisationLabels,
    ) -> Chem.Mol:
        return RDKitHelper.replace_with_placeholder(
            mol=residue.mol,
            index_to_replace=residue.get_polymerisation_atom(site.other()).idx,
            placeholder_map_num=MapLabels.INACTIVE_END,
        )

    def _set_initial_ends(
        self,
        residue: AdditionPolymerResidue,
        site: PolymerisationLabels,
    ) -> None:
        self._growing_end = AdditionPolymerEnd(map_num=site, residue_id=residue.id)
        self._inactive_end = AdditionPolymerEnd(
            map_num=MapLabels.INACTIVE_END, residue_id=residue.id
        )

    def _join_incoming(
        self,
        tagged_mol: Chem.Mol,
        site: PolymerisationLabels,
    ) -> Chem.Mol:
        tagged_mol = RDKitHelper.replace_map_num(
            mol=tagged_mol,
            old_map=site,
            new_map=MapLabels.INCOMING,
        )
        incoming_site = MolAtom(
            mol=tagged_mol,
            idx=RDKitHelper.get_site_idx(
                mol=tagged_mol,
                atom_num=0,
                map_num=MapLabels.INCOMING,
            ),
        )
        return RDKitHelper.single_bond_join_at_wildcard_sites(
            site1=self.growing_end_atom,
            site2=incoming_site,
        )

    def _prepare_cap_mols(
        self,
        cap: Cap,
    ) -> tuple[Chem.Mol, Chem.Mol]:
        cap_mol_growing = RDKitHelper.relabel_wildcard(
            mol=Chem.rdmolfiles.MolFromSmiles(cap.smiles),
            new_map_num=MapLabels.CAP_GROWING,
        )
        cap_mol_inactive = RDKitHelper.relabel_wildcard(
            mol=Chem.rdmolfiles.MolFromSmiles(cap.smiles),
            new_map_num=MapLabels.CAP_INACTIVE,
        )
        return cap_mol_growing, cap_mol_inactive

    def _join_caps(
        self,
        cap_mol_growing: Chem.Mol,
        cap_mol_inactive: Chem.Mol,
    ) -> Chem.Mol:
        return RDKitHelper.join_many_at_map_nums(
            mol=self.mol,
            mols_to_combine=[cap_mol_growing, cap_mol_inactive],
            pairs=[
                (self.growing_end.map_num, MapLabels.CAP_GROWING),
                (MapLabels.INACTIVE_END, MapLabels.CAP_INACTIVE),
            ],
        )

    def _make_cap_instances(
        self,
        cap: Cap,
        growing_tags: frozenset[int],
        inactive_tags: frozenset[int],
    ) -> list[ResidueInstance]:
        return [
            ResidueInstance(
                residue_id=cap.id,
                instance_number=0,
                residue_tags=growing_tags,
                residue_type=ResidueType.CAP,
            ),
            ResidueInstance(
                residue_id=cap.id,
                instance_number=1,
                residue_tags=inactive_tags,
                residue_type=ResidueType.CAP,
            ),
        ]

    def _tag_atoms(self, mol: Chem.Mol) -> tuple[frozenset[int], Chem.Mol]:
        rw = Chem.rdchem.RWMol(mol)
        tags = set()
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() != 0:
                self._tag_counter += 1
                atom.SetIntProp(RESIDUE_TAG, self._tag_counter)
                tags.add(self._tag_counter)
        return frozenset(tags), rw.GetMol()

    def _register_residue(
        self,
        residue: AdditionPolymerResidue,
        tags: frozenset[int],
        residue_type: ResidueType,
    ) -> ResidueInstance:
        instance_number = self._instance_counts[residue.id]
        self._instance_counts[residue.id] += 1
        instance = ResidueInstance(
            residue_id=residue.id,
            instance_number=instance_number,
            residue_tags=tags,
            residue_type=residue_type,
        )
        self._residue_instances.append(instance)
        return instance

    def _random_site(self) -> PolymerisationLabels:
        return random.choice([MapLabels.HEAD, MapLabels.TAIL])
