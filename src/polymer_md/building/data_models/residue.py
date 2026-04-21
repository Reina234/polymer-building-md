from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from polymer_md.building.data_models.map_labels import MapLabels
from polymer_md.core.mol_atom import MolAtom
from polymer_md.utils.rdkit_helper import RDKitHelper


@dataclass(frozen=True)
class AdditionPolymerResidue:
    residue_smiles: str
    label: str | None = None
    _mol: Chem.Mol = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        mol = self._parse_and_validate(self.residue_smiles)
        object.__setattr__(self, "_mol", mol)

    def get_polymerisation_atom(self, site: MapLabels) -> MolAtom:
        if site == MapLabels.HEAD:
            return self.head
        if site == MapLabels.TAIL:
            return self.tail
        raise ValueError(f"Unknown polymerisation site type {site}")

    @property
    def mol(self) -> Chem.Mol:
        return self._mol

    @property
    def id(self) -> str:
        return self.label if self.label is not None else self.residue_smiles

    @property
    def is_regiosymmetric(self) -> bool:
        return RDKitHelper.wildcard_ranks_equal(self._mol)

    @property
    def head(self) -> MolAtom:
        return MolAtom(
            mol=self._mol,
            idx=RDKitHelper.get_site_idx(
                mol=self._mol, atom_num=0, map_num=MapLabels.HEAD
            ),
        )

    @property
    def tail(self) -> MolAtom:
        return MolAtom(
            mol=self._mol,
            idx=RDKitHelper.get_site_idx(
                mol=self._mol, atom_num=0, map_num=MapLabels.TAIL
            ),
        )

    def _parse_and_validate(self, smiles: str) -> Chem.Mol:
        mol = RDKitHelper.mol_from_smiles(smiles=smiles)

        stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        map_nums = {a.GetAtomMapNum() for a in stars}

        if len(stars) != 2:
            raise ValueError(
                f"Residue must contain exactly 2 wildcard (*) atoms, found {len(stars)}."
            )
        if map_nums != {MapLabels.HEAD, MapLabels.TAIL}:
            raise ValueError(
                f"Wildcard atoms must be labelled [*:1] and [*:2], found map nums {map_nums}."
            )

        return mol
