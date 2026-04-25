from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.solvers.base import TransitionMatrixSolver
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.core.caps import BuiltinCap, Cap
from polymer_md.geometry.base import ConformerGenerator
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.extraction.builder import FragmentLibraryBuilder
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.pipeline import TrimerParameterisationPipeline

logger = logging.getLogger(__name__)


@dataclass
class FragmentLibraryPipeline:
    specs: list[MonomerSpec]
    output_dir: Path
    library_path: Path
    cap: Cap = field(default_factory=lambda: BuiltinCap.METHYL)
    solver: TransitionMatrixSolver = field(default_factory=ProportionalSolver)
    conformer_generator: ConformerGenerator = field(default_factory=ETKDGConformerGenerator)
    probability_threshold: float = 0.01
    charge_method: str = "bcc"

    def run(self) -> FragmentLibrary:
        logger.info("FragmentLibraryPipeline: parameterising trimers...")
        parameterised_trimers = self._run_trimer_pipeline()

        logger.info(
            "FragmentLibraryPipeline: building library from %d trimers...",
            len(parameterised_trimers),
        )
        library = FragmentLibraryBuilder().build(parameterised_trimers)

        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        library.save(self.library_path)
        logger.info("FragmentLibraryPipeline: library saved to %s", self.library_path)

        return library

    def _run_trimer_pipeline(self):
        return TrimerParameterisationPipeline(
            specs=self.specs,
            cap=self.cap,
            solver=self.solver,
            conformer_generator=self.conformer_generator,
            probability_threshold=self.probability_threshold,
            charge_method=self.charge_method,
        ).run(self.output_dir)
