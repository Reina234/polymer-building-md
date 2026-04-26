from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import parmed as pmd

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.solvers.base import TransitionMatrixSolver
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.building.trimer_builder import TrimerBuilder
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.random_polymer import RandomPolymerBuilder
from polymer_md.conversion.gromacs_files import GromacsFiles
from polymer_md.core.caps import BuiltinCap, Cap
from polymer_md.core.polymer import Polymer
from polymer_md.core.residue_instance import ResidueType
from polymer_md.geometry.base import ConformerGenerator
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.fragments.matching.resolution import MeanStrategy
from polymer_md.parameterisation.fragments.data_models.match import ResolutionStrategy
from polymer_md.parameterisation.strategies.base import MissingParameterStrategy
from polymer_md.parameterisation.strategies.strict import StrictMissingParameterStrategy
from polymer_md.parameterisation.tiler import PolymerParameterisationTiler, adjust_charge_neutrality
from polymer_md.utils.parmed_helper import StructureMolDeriver
from polymer_md.utils.topology_builder import TopologyBuilder

logger = logging.getLogger(__name__)


@dataclass
class PolymerParameterisationPipeline:
    library: FragmentLibrary
    specs: list[MonomerSpec]
    n: int
    output_dir: Path
    seed: int = 42
    cap: Cap = field(default_factory=lambda: BuiltinCap.METHYL)
    solver: TransitionMatrixSolver = field(default_factory=ProportionalSolver)
    conformer_generator: ConformerGenerator = field(default_factory=ETKDGConformerGenerator)
    resolution_strategy: ResolutionStrategy = field(default_factory=MeanStrategy)
    missing_strategies: dict[type, MissingParameterStrategy] = field(default_factory=dict)
    adjust_charge: bool = True

    def run(self) -> ParameterisedMolecule:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building %d-mer polymer...", self.n)
        polymer = self._build_polymer()

        logger.info("Embedding 3D conformer...")
        mol_3d = self.conformer_generator.embed(polymer.mol)

        logger.info("Building topology...")
        structure = TopologyBuilder.build(mol_3d)

        logger.info("Deriving RDKit mol from parmed structure...")
        derived_mol = StructureMolDeriver.derive(structure)

        logger.info("Building polymer atom metadata...")
        polymer_atom_metadata = self._build_polymer_atom_metadata(polymer)

        logger.info("Tiling parameters...")
        tiler = PolymerParameterisationTiler(
            library=self.library,
            resolution_strategy=self.resolution_strategy,
            missing_strategies=self.missing_strategies,
        )
        tiler.tile(structure, derived_mol, polymer_atom_metadata)

        if self.adjust_charge:
            logger.info("Adjusting charge neutrality...")
            adjust_charge_neutrality(structure)

        logger.info("Saving GROMACS files...")
        gromacs_files = self._save_gromacs(structure)

        logger.info("PolymerParameterisationPipeline complete.")
        return ParameterisedMolecule(structure=structure, mol=mol_3d, source=gromacs_files)

    def _build_polymer(self) -> Polymer:
        residues = {spec.residue_id: spec.residue for spec in self.specs}
        matrix = self.solver.solve(self.specs)
        builder = RandomPolymerBuilder(residues=residues, matrix=matrix, cap=self.cap)
        rng = np.random.default_rng(self.seed)
        return builder.build(self.n, rng)

    @staticmethod
    def _build_polymer_atom_metadata(polymer: Polymer) -> dict[int, tuple[str, int]]:
        metadata: dict[int, tuple[str, int]] = {}
        for instance in polymer.residue_instances:
            if instance.residue_type == ResidueType.CAP:
                continue
            PolymerParameterisationPipeline._add_instance_metadata(polymer, instance, metadata)
        return metadata

    @staticmethod
    def _add_instance_metadata(
        polymer: Polymer,
        instance,
        metadata: dict[int, tuple[str, int]],
    ) -> None:
        heavy_indices = PolymerParameterisationPipeline._heavy_atom_indices(polymer, instance)
        for position, idx in enumerate(sorted(heavy_indices)):
            metadata[idx] = (instance.residue_id, position)

    @staticmethod
    def _heavy_atom_indices(polymer: Polymer, instance) -> frozenset[int]:
        return frozenset(
            idx for idx in instance.atom_indices(polymer.mol)
            if polymer.mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        )

    def _save_gromacs(self, structure: pmd.Structure) -> GromacsFiles:
        name = f"polymer_{self.n}mer"
        gro_path = self.output_dir / f"{name}.gro"
        top_path = self.output_dir / f"{name}.top"
        itp_path = self.output_dir / f"{name}.itp"
        structure.save(str(gro_path), overwrite=True)
        structure.save(str(top_path), overwrite=True)
        structure.save(str(itp_path), format="GROMACS", overwrite=True)
        return GromacsFiles(itp=itp_path, gro=gro_path, top=top_path)
