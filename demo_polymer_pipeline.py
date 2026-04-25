"""
Demo: PolymerParameterisationPipeline (Orchestrator 2)

Builds and parameterises a random polystyrene 10-mer using a pre-built library.
Requires: OBabel and acpype installed on PATH, and a library at output/demo_library/library.json
(run demo_fragment_library_pipeline.py first).

Usage:
    python demo_polymer_pipeline.py
"""
import logging
from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

LIBRARY_PATH = Path("output/demo_library/library.json")


def main() -> None:
    print("=== Polymer Parameterisation Pipeline Demo ===\n")

    if not LIBRARY_PATH.exists():
        print(f"Library not found at {LIBRARY_PATH}. Run demo_fragment_library_pipeline.py first.")
        return

    print(f"Loading library from {LIBRARY_PATH}...")
    library = FragmentLibrary.load(LIBRARY_PATH)
    print(f"  {len(library.records)} records, {len(library.atom_metadata)} fragment patterns\n")

    styrene_residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    specs = [MonomerSpec(residue=styrene_residue, weight=1.0)]

    pipeline = PolymerParameterisationPipeline(
        library=library,
        specs=specs,
        n=10,
        output_dir=Path("output/demo_polymer"),
        seed=42,
        conformer_generator=ETKDGConformerGenerator(use_uff=False),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
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
