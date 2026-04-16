from __future__ import annotations

import logging
from pathlib import Path

from openbabel import pybel

from polymer_md.conversion.base import Converter, handles
from polymer_md.conversion.file_formats import FileFormats
from polymer_md.core.molecule_input import MoleculeInput
from polymer_md.conversion.registry import register_converter
from polymer_md.utils.file import FileHelper

logger = logging.getLogger(__name__)


@register_converter
class OBabelConverter(Converter):
    @handles(MoleculeInput, FileFormats.MOL2)
    def _molecule_to_mol2(
        self,
        source: MoleculeInput,
        output_dir: Path,
        output_name: str,
        overwrite: bool,
    ) -> Path:
        mol = pybel.readstring("smi", source.smiles)
        return self._write_mol(mol, FileFormats.MOL2, output_dir, output_name, overwrite)

    @handles(FileFormats.PDB, FileFormats.MOL2)
    def _pdb_to_mol2(
        self,
        source: Path,
        output_dir: Path,
        output_name: str,
        overwrite: bool,
    ) -> Path:
        mol = self._read_mol_from_path(source)
        return self._write_mol(mol, FileFormats.MOL2, output_dir, output_name, overwrite)

    @handles(FileFormats.PDB, FileFormats.GRO)
    def _pdb_to_gro(
        self,
        source: Path,
        output_dir: Path,
        output_name: str,
        overwrite: bool,
    ) -> Path:
        mol = self._read_mol_from_path(source)
        return self._write_mol(mol, FileFormats.GRO, output_dir, output_name, overwrite)

    @staticmethod
    def _read_mol_from_path(source: Path) -> pybel.Molecule:
        input_format = FileHelper.get_suffix_type(source)
        return list(pybel.readfile(input_format, str(source)))[0]

    @staticmethod
    def _write_mol(
        mol: pybel.Molecule,
        output_format: FileFormats,
        output_dir: Path,
        output_name: str,
        overwrite: bool,
    ) -> Path:
        output_path = FileHelper.construct_path(
            path_dir=output_dir, stem=output_name, suffix=output_format
        )
        mol.write(str(output_format), str(output_path), overwrite=overwrite)
        logger.info("OBabel output saved to %s", output_path)
        return output_path
