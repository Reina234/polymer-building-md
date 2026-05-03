"""
Demo: PolymerParameterisationPipeline

Builds and parameterises a random polystyrene 10-mer. The pipeline builds the
fragment library on-demand from the trimers required by the actual polymer sequence.

Requires: OBabel and acpype installed on PATH.

Usage:
    conda run -n md_engines python demo_polymer_pipeline.py
"""
import logging
from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")


def main() -> None:
    print("=== Polymer Parameterisation Pipeline Demo ===\n")

    styrene_residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    specs = [MonomerSpec(residue=styrene_residue, weight=1.0)]

    pipeline = PolymerParameterisationPipeline(
        specs=specs,
        n=10,
        output_dir=Path("output/demo_polymer"),
        seed=42,
        conformer_generator=ETKDGConformerGenerator(use_uff=False, use_random_coords=True),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
        charge_method="bcc",
    )

    print("Building and parameterising 10-mer polystyrene...")
    result = pipeline.run()

    total_charge = sum(a.charge for a in result.structure.atoms)
    print(f"\nParameterised molecule:")
    print(f"  Atoms:         {len(result.structure.atoms)}")
    print(f"  Bonds:         {len(result.structure.bonds)}")
    print(f"  Angles:        {len(result.structure.angles)}")
    print(f"  Dihedrals:     {len(result.structure.dihedrals)}")
    print(f"  Total charge:  {total_charge:.4f} e")
    print(f"\nGRO: {result.source.gro}")
    print(f"TOP: {result.source.top}")
    print(f"ITP: {result.source.itp}")


if __name__ == "__main__":
    main()
