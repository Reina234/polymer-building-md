from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm  # type: ignore[import-untyped]

import parmed as pmd

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.residue import AdditionPolymerResidue
from polymer_md.building.data_models.trimer import Orientation, TrimerResult
from polymer_md.building.solvers.base import TransitionMatrixSolver
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.building.trimer_builder import TrimerBuilder
from polymer_md.conversion.acpype import AcpypeConverter
from polymer_md.conversion.file_formats import FileFormats
from polymer_md.conversion.gromacs_files import GromacsFiles
from polymer_md.conversion.obabel import OBabelConverter
from polymer_md.core.caps import BuiltinCap, Cap
from polymer_md.geometry.base import ConformerGenerator
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.data_models.parameterised_trimer import (
    ParameterisedTrimer,
)
from polymer_md.utils.rdkit_helper import RDKitHelper

logger = logging.getLogger(__name__)


@dataclass
class TrimerParameterisationPipeline:
    specs: list[MonomerSpec]
    cap: Cap = field(default_factory=lambda: BuiltinCap.METHYL)
    solver: TransitionMatrixSolver = field(default_factory=ProportionalSolver)
    conformer_generator: ConformerGenerator = field(
        default_factory=ETKDGConformerGenerator
    )
    probability_threshold: float = 0.01
    charge_method: str = "bcc"

    def run(self, output_dir: Path) -> list[ParameterisedTrimer]:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Pipeline starting: %d monomer specs, cap=%s, charge_method=%s",
            len(self.specs),
            self.cap.id,
            self.charge_method,
        )

        residues = self._build_residue_map()
        matrix = self._solve_transition_matrix()

        logger.info(
            "Building trimers for %d monomers: %s",
            len(self.specs),
            [spec.residue_id for spec in self.specs],
        )
        all_trimers = TrimerBuilder(
            residues=residues, matrix=matrix, cap=self.cap
        ).build_all()

        selected = self._filter_by_probability(all_trimers)
        dropped = len(all_trimers) - len(selected)
        logger.info(
            "%d trimers built: %d selected, %d dropped (p < %.4f)",
            len(all_trimers),
            len(selected),
            dropped,
            self.probability_threshold,
        )

        results = []
        for trimer in tqdm(selected, desc="Parameterizing trimers", unit="trimer"):
            logger.info(
                "Parameterizing %s-%s-%s (%s) p=%.4f",
                trimer.left_id,
                trimer.central_id,
                trimer.right_id,
                trimer.orientation.name,
                trimer.probability,
            )
            results.append(self._parameterise_trimer(trimer, output_dir))

        logger.info("Pipeline complete: %d trimers parameterised.", len(results))
        return results

    def _build_residue_map(self) -> dict[str, AdditionPolymerResidue]:
        residues = {spec.residue_id: spec.residue for spec in self.specs}
        logger.info("Residue map: %s", list(residues.keys()))
        return residues

    def _solve_transition_matrix(self):
        logger.info("Solving transition matrix with %s...", type(self.solver).__name__)
        matrix = self.solver.solve(self.specs)
        logger.info("Transition matrix solved: %d sites.", matrix.n_sites)
        return matrix

    def _filter_by_probability(self, trimers: list[TrimerResult]) -> list[TrimerResult]:
        return [t for t in trimers if t.probability >= self.probability_threshold]

    def _parameterise_trimer(
        self, trimer: TrimerResult, output_dir: Path
    ) -> ParameterisedTrimer:
        name = self._trimer_name(trimer)
        trimer_dir = output_dir / name
        trimer_dir.mkdir(parents=True, exist_ok=True)

        logger.info("  [%s] Embedding 3D conformer...", name)
        mol_3d = self.conformer_generator.embed(trimer.mol)

        sdf_path = trimer_dir / f"{name}.sdf"
        RDKitHelper.write_sdf(mol_3d, sdf_path)
        logger.info("  [%s] SDF written: %s", name, sdf_path)

        logger.info("  [%s] Converting SDF → MOL2 (OBabel)...", name)
        mol2_path = OBabelConverter().convert(
            source=sdf_path,
            output_type=FileFormats.MOL2,
            output_dir=trimer_dir,
            output_name=name,
        )
        logger.info("  [%s] MOL2 written: %s", name, mol2_path)

        logger.info(
            "  [%s] Running acpype (charge_method=%s)...", name, self.charge_method
        )
        gromacs_files = AcpypeConverter(charge_method=self.charge_method).convert(
            source=mol2_path,
            output_type=GromacsFiles,
            output_dir=trimer_dir,
            output_name=name,
            overwrite=True,
        )
        logger.info("  [%s] GROMACS files: %s", name, gromacs_files.gro)

        logger.info("  [%s] Loading parmed structure...", name)
        structure = pmd.load_file(str(gromacs_files.top), xyz=str(gromacs_files.gro))
        logger.info("  [%s] Done. Atoms in structure: %d", name, len(structure.atoms))

        return ParameterisedTrimer(
            trimer_result=trimer,
            gromacs_files=gromacs_files,
            structure=structure,
        )

    @staticmethod
    def _trimer_name(trimer: TrimerResult) -> str:
        orientation_label = "HI" if trimer.orientation == Orientation.HEAD_IN else "TI"
        smiles = RDKitHelper.canonical_smiles_stripped(trimer.mol)
        smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:6]
        return (
            f"{trimer.left_id}_{trimer.central_id}_{trimer.right_id}"
            f"_{orientation_label}_{smiles_hash}"
        )
