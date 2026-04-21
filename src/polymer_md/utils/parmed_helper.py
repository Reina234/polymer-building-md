from __future__ import annotations

import tempfile

import numpy as np
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


class StructureMolDeriver:
    @staticmethod
    def derive(structure: pmd.Structure) -> Chem.Mol:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temporary_file:
            pdb_path = temporary_file.name
        structure.save(pdb_path, overwrite=True)
        raw = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        if raw is None:
            raise ValueError(f"RDKit could not parse parmed-written PDB: {pdb_path}")
        rdDetermineBonds.DetermineBonds(raw, charge=0)
        Chem.SanitizeMol(raw)
        return raw


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
