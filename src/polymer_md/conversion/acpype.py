from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from polymer_md.conversion.base import Converter, handles
from polymer_md.conversion.file_formats import FileFormats
from polymer_md.conversion.gromacs_files import GromacsFiles
from polymer_md.conversion.registry import register_converter

logger = logging.getLogger(__name__)


@register_converter
@dataclass
class AcpypeConverter(Converter):
    charge_method: str = "bcc"

    @handles(FileFormats.MOL2, GromacsFiles)
    def _mol2_to_gromacs(
        self,
        source: Path,
        output_dir: Path,
        output_name: str,
        overwrite: bool,
    ) -> GromacsFiles:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_output_dir(output_dir, output_name, overwrite)
        result = self._run_acpype(source, output_dir, output_name, self.charge_method)
        self._log_result(result=result)
        return self._collect_outputs(output_dir, output_name)

    @staticmethod
    def _prepare_output_dir(
        output_dir: Path, output_name: str, overwrite: bool
    ) -> None:
        acpype_dir = output_dir / f"{output_name}.acpype"
        if acpype_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {acpype_dir}. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(acpype_dir)

    @staticmethod
    def _run_acpype(
        source: Path,
        output_dir: Path,
        output_name: str,
        charge_method: str,
    ) -> subprocess.CompletedProcess:
        command = [
            "acpype",
            "-i",
            str(source.resolve()),
            "-b",
            output_name,
            "-o",
            "gmx",
            "-c",
            charge_method,
        ]
        logger.info("Running: %s", " ".join(command))
        result = subprocess.run(
            command, cwd=output_dir, capture_output=True, text=True, check=False
        )
        return result

    @staticmethod
    def _log_result(result: subprocess.CompletedProcess) -> None:
        if result.stdout:
            logger.info("acpype stdout:\n%s", result.stdout)
        if result.stderr:
            logger.info("acpype stderr:\n%s", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"acpype failed (exit code {result.returncode}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    @staticmethod
    def _collect_outputs(output_dir: Path, output_name: str) -> GromacsFiles:
        acpype_dir = output_dir / f"{output_name}.acpype"
        return GromacsFiles(
            itp=acpype_dir / f"{output_name}_GMX.itp",
            gro=acpype_dir / f"{output_name}_GMX.gro",
            top=acpype_dir / f"{output_name}_GMX.top",
        )
