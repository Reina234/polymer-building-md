"""
Parameterise a random polymer and write GROMACS files.

Swap out the monomer SMILES, labels, weights, and n to parameterise any
addition polymer. For copolymers add more MonomerSpec entries.

Run:
    PATH="/Users/reinazheng/miniconda3/envs/md_engines/bin:$PATH" uv run python examples/parameterise_polymer.py
"""
from __future__ import annotations

from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy

specs = [
    MonomerSpec(
        residue=MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S")),
        weight=1.0,
    ),
]

pipeline = PolymerParameterisationPipeline(
    specs=specs,
    n=10,
    output_dir=Path("output/polystyrene_10mer"),
    seed=42,
    solver=ProportionalSolver(ht_fraction=0.95),
    conformer_generator=ETKDGConformerGenerator(use_uff=False, use_random_coords=True),
    missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
    adjust_charge=True,
    charge_method="bcc",
)

result = pipeline.run()

total_charge = sum(a.charge for a in result.structure.atoms)
print(f"Atoms:        {len(result.structure.atoms)}")
print(f"Bonds:        {len(result.structure.bonds)}")
print(f"Total charge: {total_charge:.4f} e")
print(f"GRO: {result.source.gro}")
print(f"TOP: {result.source.top}")
