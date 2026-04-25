from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import parmed as pmd
from rdkit import Chem

from polymer_md.conversion.acpype import AcpypeConverter
from polymer_md.conversion.file_formats import FileFormats
from polymer_md.conversion.gromacs_files import GromacsFiles
from polymer_md.conversion.obabel import OBabelConverter
from polymer_md.geometry.base import ConformerGenerator
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
)
from polymer_md.parameterisation.strategies.base import (
    MissingParameterError,
    StrategyContext,
)
from polymer_md.utils.rdkit_helper import RDKitHelper


@dataclass
class MinimalMoleculeStrategy:
    """Parameterises a minimal molecule extracted from the polymer context.

    Intended for bond, angle, and dihedral parameters that are absent from the
    library. The strategy cuts out the atoms defining the missing term plus their
    immediate neighbours, caps dangling valences with implicit H, runs the tiny
    fragment through OBabel + acpype, and reads the specific parameter back out.
    """

    conformer_generator: ConformerGenerator = field(
        default_factory=ETKDGConformerGenerator
    )
    charge_method: str = "bcc"

    def resolve(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> float:
        if isinstance(parameter, AtomParameter):
            raise MissingParameterError(
                f"MinimalMoleculeStrategy handles bond/angle/dihedral parameters, "
                f"not {type(parameter).__name__}. "
                f"Use ResiduePositionStrategy or NeighbourhoodSMARTSStrategy for atom parameters."
            )
        mol, old_to_new = self._extract_minimal_mol(context.derived_mol, global_indices)
        with tempfile.TemporaryDirectory() as tmp:
            structure = self._parameterise(mol, Path(tmp))
        local_indices = tuple(old_to_new[i] for i in global_indices)
        return self._read_parameter(structure, local_indices, parameter)

    @staticmethod
    def _extract_minimal_mol(
        derived_mol: Chem.Mol,
        global_indices: tuple[int, ...],
    ) -> tuple[Chem.Mol, dict[int, int]]:
        atom_set = set(global_indices)
        for idx in global_indices:
            for nbr in derived_mol.GetAtomWithIdx(idx).GetNeighbors():
                atom_set.add(nbr.GetIdx())

        sorted_atoms = sorted(atom_set)
        old_to_new: dict[int, int] = {old: new for new, old in enumerate(sorted_atoms)}

        rw = Chem.RWMol()
        for old_idx in sorted_atoms:
            src = derived_mol.GetAtomWithIdx(old_idx)
            rw.AddAtom(Chem.Atom(src.GetAtomicNum()))

        for bond in derived_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i in atom_set and j in atom_set:
                rw.AddBond(old_to_new[i], old_to_new[j], bond.GetBondType())

        Chem.SanitizeMol(rw)
        return Chem.AddHs(rw), old_to_new

    def _parameterise(self, mol: Chem.Mol, tmp_dir: Path) -> pmd.Structure:
        mol_3d = self.conformer_generator.embed(mol)
        sdf_path = tmp_dir / "minimal.sdf"
        RDKitHelper.write_sdf(mol_3d, sdf_path)
        mol2_path = OBabelConverter().convert(
            source=sdf_path,
            output_type=FileFormats.MOL2,
            output_dir=tmp_dir,
            output_name="minimal",
        )
        gromacs_files = AcpypeConverter(charge_method=self.charge_method).convert(
            source=mol2_path,
            output_type=GromacsFiles,
            output_dir=tmp_dir,
            output_name="minimal",
            overwrite=True,
        )
        return pmd.load_file(str(gromacs_files.top), xyz=str(gromacs_files.gro))

    @staticmethod
    def _read_parameter(
        structure: pmd.Structure,
        local_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
    ) -> float:
        if isinstance(parameter, BondParameter):
            return MinimalMoleculeStrategy._bond_value(
                structure, local_indices, parameter
            )
        if isinstance(parameter, AngleParameter):
            return MinimalMoleculeStrategy._angle_value(
                structure, local_indices, parameter
            )
        if isinstance(parameter, DihedralParameter):
            return MinimalMoleculeStrategy._dihedral_value(
                structure, local_indices, parameter
            )
        raise MissingParameterError(
            f"Unsupported parameter type: {type(parameter).__name__}"
        )

    @staticmethod
    def _bond_value(
        structure: pmd.Structure,
        atom_indices: tuple[int, ...],
        parameter: BondParameter,
    ) -> float:
        index_set = frozenset(atom_indices)
        for bond in structure.bonds:
            if frozenset({bond.atom1.idx, bond.atom2.idx}) == index_set:
                if parameter == BondParameter.FORCE_CONSTANT:
                    return float(bond.type.k)
                return float(bond.type.req)
        raise MissingParameterError(
            f"Bond not found between atoms {atom_indices} in minimal structure"
        )

    @staticmethod
    def _angle_value(
        structure: pmd.Structure,
        atom_indices: tuple[int, ...],
        parameter: AngleParameter,
    ) -> float:
        index_set = frozenset(atom_indices)
        for angle in structure.angles:
            if (
                frozenset({angle.atom1.idx, angle.atom2.idx, angle.atom3.idx})
                == index_set
            ):
                if parameter == AngleParameter.FORCE_CONSTANT:
                    return float(angle.type.k)
                return float(angle.type.theteq)
        raise MissingParameterError(
            f"Angle not found for atoms {atom_indices} in minimal structure"
        )

    @staticmethod
    def _dihedral_value(
        structure: pmd.Structure,
        atom_indices: tuple[int, ...],
        parameter: DihedralParameter,
    ) -> float:
        index_set = frozenset(atom_indices)
        for dihedral in structure.dihedrals:
            if dihedral.improper:
                continue
            indices = frozenset(
                {
                    dihedral.atom1.idx,
                    dihedral.atom2.idx,
                    dihedral.atom3.idx,
                    dihedral.atom4.idx,
                }
            )
            if indices == index_set:
                if parameter == DihedralParameter.FORCE_CONSTANT:
                    return float(dihedral.type.phi_k)
                if parameter == DihedralParameter.PHASE:
                    return float(dihedral.type.phase)
                return float(dihedral.type.per)
        raise MissingParameterError(
            f"Dihedral not found for atoms {atom_indices} in minimal structure"
        )
