from __future__ import annotations

import copy
import logging
from pathlib import Path

import parmed as pmd

from polymer_md.conversion.base import Converter, handles
from polymer_md.conversion.file_formats import FileFormats
from polymer_md.conversion.registry import register_converter
from polymer_md.utils.file import FileHelper

logger = logging.getLogger(__name__)


@register_converter
class ParmEdConverter(Converter):
    def __init__(self, bounding_box: list[int] | None = None) -> None:
        self._bounding_box = bounding_box

    @handles(FileFormats.PDB, FileFormats.GRO)
    def _pdb_to_gro(
        self,
        source: Path,
        output_dir: Path,
        output_name: str,
        overwrite: bool,
    ) -> Path:
        struct = copy.deepcopy(pmd.load_file(str(source)))
        if self._bounding_box is not None:
            struct.box = self._bounding_box
        output_path = FileHelper.construct_path(
            path_dir=output_dir, stem=output_name, suffix=FileFormats.GRO
        )
        struct.save(str(output_path), overwrite=overwrite)
        logger.info("ParmEd output saved to %s", output_path)
        return output_path
