import copy
import logging
from pathlib import Path
from typing import List, Literal, Optional, get_args

import parmed as pmd

from polymer_md.conversion.file_formats import FileFormats
from polymer_md.utils.file import FileHelper, PathType

logger = logging.getLogger(__name__)
ParmEdInputs = Literal[FileFormats.PDB]
ParmEdOutputs = Literal[FileFormats.GRO]
PARMED_INPUTS: set[str] = set(get_args(ParmEdInputs))
PARMED_OUTPUTS: set[str] = set(get_args(ParmEdOutputs))


class ParmEdConverter:
    default_dir = Path("parmed_outputs/")

    def __init__(self, item_to_convert: PathType) -> None:
        self._struct = self._get_struct_from_path(input_path=item_to_convert)
        self.default_name = Path(item_to_convert).stem

    def convert(
        self,
        output_format: ParmEdOutputs,
        output_dir: Optional[PathType] = None,
        output_name: Optional[str] = None,
        bounding_box: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> Path:
        logger.info("Conversion started")
        output_dir = output_dir or self.default_dir
        output_name = output_name or self.default_name
        output_path = FileHelper.construct_path(
            path_dir=output_dir, stem=output_name, suffix=output_format
        )

        struct = copy.deepcopy(self._struct)
        if bounding_box:
            struct.box = bounding_box

        struct.save(str(output_path), overwrite=overwrite)

        return output_path

    def _get_struct_from_path(self, input_path: PathType):
        return pmd.load_file(str(input_path))
