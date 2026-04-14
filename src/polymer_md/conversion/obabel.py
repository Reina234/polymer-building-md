import logging
from pathlib import Path
from typing import Literal, Optional, Union, get_args

from openbabel import pybel

from polymer_md.conversion.file_formats import FileFormats
from polymer_md.core.monomer import Monomer
from polymer_md.utils.file import FileHelper, PathType

logger = logging.getLogger(__name__)
OBabelInputs = Literal[FileFormats.PDB]
OBabelOutputs = Literal[FileFormats.MOL2, FileFormats.GRO]
OBABEL_INPUTS: set[str] = set(get_args(OBabelInputs))
OBABEL_OUTPUTS: set[str] = set(get_args(OBabelOutputs))


class OBabelConverter:
    default_dir = Path("obabel_outputs/")

    def __init__(self, item_to_convert: Union[PathType, Monomer]) -> None:
        self._mol: pybel.Molecule = self._get_mol(item_to_convert=item_to_convert)
        self.default_name: str = self._get_default_name(item_to_convert=item_to_convert)

    def _get_mol(self, item_to_convert: Union[PathType, Monomer]) -> pybel.Molecule:
        if isinstance(item_to_convert, Monomer):
            return self._get_mol_object_from_smiles(smiles=item_to_convert.smiles)
        if isinstance(item_to_convert, PathType):
            return self._get_mol_object_from_path(input_path=item_to_convert)

        raise ValueError(f"[UNSUPPORTED_INPUT_TYPE]:{type(item_to_convert)}")

    def _get_default_name(self, item_to_convert: Union[PathType, Monomer]) -> str:
        if isinstance(item_to_convert, Monomer):
            return item_to_convert.label or item_to_convert.smiles
        if isinstance(item_to_convert, PathType):
            return Path(item_to_convert).stem

        raise ValueError(f"[UNSUPPORTED_INPUT_TYPE]:{type(item_to_convert)}")

    def convert(
        self,
        output_format: OBabelOutputs,
        output_dir: Optional[PathType] = None,
        output_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        logger.info("Conversion started")
        output_dir = output_dir or self.default_dir
        output_name = output_name or self.default_name
        output_path = FileHelper.construct_path(
            path_dir=output_dir, stem=output_name, suffix=output_format
        )
        self._save_mol_object(
            self._mol,
            output_path=output_path,
            overwrite=overwrite,
        )

        return output_path

    def _save_mol_object(
        self,
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

    def _get_mol_object_from_path(self, input_path: PathType) -> pybel.Molecule:
        input_format = FileHelper.safe_get_suffix_type(
            path=input_path, supported_suffixes=OBABEL_INPUTS
        )
        pybel_objects = list(pybel.readfile(input_format, str(input_path)))
        logger.info(
            "OBabel object read from %s",
            input_path,
        )
        return pybel_objects[0]

    def _get_mol_object_from_smiles(self, smiles: str) -> pybel.Molecule:
        mol = pybel.readstring("smi", smiles)
        logger.info(
            "OBabel object read from %s",
            smiles,
        )
        return mol
