from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from polymer_md.building.sites import PolymerisationSite
from polymer_md.utils.rdkit_helper import RDKitHelper


@dataclass(frozen=True)
class AdditionPolymerResidue:
    residue_smiles: str
    label: str | None = None
    _mol: Chem.Mol = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        mol = self._parse_and_validate(self.residue_smiles)
        object.__setattr__(self, "_mol", mol)

    @property
    def mol(self) -> Chem.Mol:
        return self._mol

    @property
    def id(self) -> str:
        return self.label if self.label is not None else self.residue_smiles

    @property
    def site_1_idx(self) -> int:
        return RDKitHelper.get_site_idx(mol=self._mol, atom_num=0, map_num=1)

    @property
    def site_2_idx(self) -> int:
        return RDKitHelper.get_site_idx(mol=self._mol, atom_num=0, map_num=2)

    def _parse_and_validate(self, smiles: str) -> Chem.Mol:
        mol = RDKitHelper.mol_from_smiles(smiles=smiles)

        stars = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        map_nums = {a.GetAtomMapNum() for a in stars}

        if len(stars) != len(PolymerisationSite):
            raise ValueError(
                f"Residue must contain exactly 2 wildcard (*) atoms, found {len(stars)}."
            )
        if map_nums != set(PolymerisationSite):
            raise ValueError(
                f"Wildcard atoms must be labelled [*:1] and [*:2], found map nums {map_nums}."
            )

        return mol
