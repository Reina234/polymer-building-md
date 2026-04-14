import logging
from pathlib import Path
from typing import Literal, get_args

from openbabel import pybel

from polymer_md.conversion.file_formats import FileFormats
from polymer_md.utils.file import FileHelper, PathType

logger = logging.getLogger(__name__)
ObabelInputs = Literal[FileFormats.PDB]
ObabelOutputs = Literal[FileFormats.MOL2, FileFormats.GRO]
OBABEL_INPUTS: set[str] = set(get_args(ObabelInputs))
OBABEL_OUTPUTS: set[str] = set(get_args(ObabelOutputs))


class ObabelConverter:
    @classmethod
    def convert_smiles(
        cls,
        smiles: str,
        output_dir: PathType,
        output_name: str,
        output_format: ObabelOutputs,
        overwrite: bool = False,
    ) -> Path:
        logger.info("Conversion started")
        output_path = FileHelper.construct_path(
            path_dir=output_dir, stem=output_name, suffix=output_format
        )
        cls._save_mol_object(
            cls._get_mol_object_from_smiles(smiles=smiles),
            output_path=output_path,
            overwrite=overwrite,
        )

        return output_path

    @classmethod
    def convert_file(
        cls,
        input_path: PathType,
        output_dir: PathType,
        output_name: str,
        output_format: ObabelOutputs,
        overwrite: bool = False,
    ) -> Path:
        logger.info("Conversion started")
        output_path = FileHelper.construct_path(
            path_dir=output_dir, stem=output_name, suffix=output_format
        )
        cls._save_mol_object(
            cls._get_mol_object_from_path(input_path=input_path),
            output_path=output_path,
            overwrite=overwrite,
        )

        return output_path

    @classmethod
    def _save_mol_object(
        cls,
        mol_object: pybel.Molecule,
        output_path: PathType,
        overwrite: bool = False,
    ) -> None:
        output_format = FileHelper.safe_get_suffix_type(
            path=output_path, supported_suffixes=OBABEL_OUTPUTS
        )
        mol_object.write(output_format, str(output_path), overwrite=overwrite)
        logger.info(
            "Output saved to %s",
            output_path,
        )

    @classmethod
    def _get_mol_object_from_path(cls, input_path: PathType) -> pybel.Molecule:
        input_format = FileHelper.safe_get_suffix_type(
            path=input_path, supported_suffixes=OBABEL_INPUTS
        )
        pybel_objects = list(pybel.readfile(input_format, str(input_path)))
        logger.info(
            "OBabel object read from %s",
            input_path,
        )
        return pybel_objects[0]

    @classmethod
    def _get_mol_object_from_smiles(cls, smiles: str) -> pybel.Molecule:
        mol = pybel.readstring("smi", smiles)
        logger.info(
            "OBabel object read from %s",
            smiles,
        )
        return mol
