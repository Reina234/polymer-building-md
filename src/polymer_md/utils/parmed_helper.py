from __future__ import annotations

import numpy as np
import parmed as pmd
from parmed.topologyobjects import DihedralTypeList
from rdkit import Chem


def mol_from_structure(structure: pmd.Structure) -> Chem.Mol:
    """Build a bare RDKit mol from parmed atom/bond topology (no 3D coords needed)."""
    edit = Chem.RWMol()
    for atom in structure.atoms:
        edit.AddAtom(Chem.Atom(atom.atomic_number))
    for bond in structure.bonds:
        edit.AddBond(bond.atom1.idx, bond.atom2.idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(edit)
    except Chem.rdchem.MolSanitizeException:
        pass
    return edit.GetMol()


class StructureMolDeriver:
    @staticmethod
    def derive(structure: pmd.Structure, mol_3d: Chem.Mol) -> Chem.Mol:
        mol3d_to_parmed = CoordinateCrosswalk.map_mol3d_to_parmed(mol_3d, structure)
        edit = Chem.RWMol()
        for atom in structure.atoms:
            edit.AddAtom(Chem.Atom(atom.atomic_number))
        for bond in mol_3d.GetBonds():
            i_pmd = mol3d_to_parmed[bond.GetBeginAtomIdx()]
            j_pmd = mol3d_to_parmed[bond.GetEndAtomIdx()]
            edit.AddBond(i_pmd, j_pmd, bond.GetBondType())
        Chem.SanitizeMol(edit)
        return edit.GetMol()


class CoordinateCrosswalk:
    @staticmethod
    def map_mol3d_to_parmed(
        mol_3d: Chem.Mol,
        structure: pmd.Structure,
    ) -> dict[int, int]:
        mol_coords = CoordinateCrosswalk._mol3d_coords(mol_3d)
        parmed_coords = CoordinateCrosswalk._parmed_coords(structure)
        return {
            mol_index: CoordinateCrosswalk._nearest_parmed_index(mol_coord, parmed_coords)
            for mol_index, mol_coord in enumerate(mol_coords)
        }

    @staticmethod
    def _mol3d_coords(mol_3d: Chem.Mol) -> np.ndarray:
        conformer = mol_3d.GetConformer()
        return np.array(
            [list(conformer.GetAtomPosition(i)) for i in range(mol_3d.GetNumAtoms())]
        )

    @staticmethod
    def _parmed_coords(structure: pmd.Structure) -> np.ndarray:
        return np.array([[atom.xx, atom.xy, atom.xz] for atom in structure.atoms])

    @staticmethod
    def _nearest_parmed_index(mol_coord: np.ndarray, parmed_coords: np.ndarray) -> int:
        distances = np.linalg.norm(parmed_coords - mol_coord, axis=1)
        return int(np.argmin(distances))


class ParmedTypeResolver:
    @staticmethod
    def bond_type(bond: pmd.Bond) -> pmd.BondType:
        if bond.type is None:
            raise ValueError(
                f"Bond between atoms {bond.atom1.idx} and {bond.atom2.idx} has no type assigned"
            )
        return bond.type

    @staticmethod
    def angle_type(angle: pmd.Angle) -> pmd.AngleType:
        if angle.type is None:
            raise ValueError(
                f"Angle among atoms {angle.atom1.idx}, {angle.atom2.idx}, {angle.atom3.idx} "
                f"has no type assigned"
            )
        return angle.type

    @staticmethod
    def dihedral_types(dihedral: pmd.Dihedral) -> list[pmd.DihedralType]:
        dtype = dihedral.type
        if dtype is None:
            raise ValueError(
                f"Dihedral among atoms {dihedral.atom1.idx}, {dihedral.atom2.idx}, "
                f"{dihedral.atom3.idx}, {dihedral.atom4.idx} has no type assigned"
            )
        if isinstance(dtype, DihedralTypeList):
            return list(dtype)
        return [dtype]

    @staticmethod
    def dihedral_type(dihedral: pmd.Dihedral) -> pmd.DihedralType:
        return ParmedTypeResolver.dihedral_types(dihedral)[0]
