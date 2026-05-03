"""
Demo: FragmentLibraryPipeline

Pre-builds a fragment parameter library for all polystyrene trimers above a
probability threshold and saves it to disk for inspection.

Note: For end-to-end polymer parameterisation you do not need to run this
separately — PolymerParameterisationPipeline builds the library on-demand from
the trimers required by the actual polymer sequence. This demo is useful for
inspecting the library contents or pre-caching trimers.

Requires: OBabel and acpype installed on PATH.

Usage:
    conda run -n md_engines python demo_fragment_library_pipeline.py
"""
import logging
from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragment_library_pipeline import FragmentLibraryPipeline
from polymer_md.parameterisation.fragments.library import FragmentLibrary

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")


def main() -> None:
    print("=== Fragment Library Pipeline Demo ===\n")

    styrene_residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    specs = [MonomerSpec(residue=styrene_residue, weight=1.0)]

    pipeline = FragmentLibraryPipeline(
        specs=specs,
        output_dir=Path("output/demo_library/trimers"),
        library_path=Path("output/demo_library/library.json"),
        conformer_generator=ETKDGConformerGenerator(use_uff=False, use_random_coords=True),
        probability_threshold=0.0,
        charge_method="bcc",
    )

    print("Running trimer parameterisation and building library...")
    library = pipeline.run()

    print(f"\nLibrary built: {len(library.records)} parameter records")
    print(f"Atom metadata entries: {sum(len(m) for m in library.atom_metadata.values())}")
    print(f"\nLibrary saved to: {pipeline.library_path}")

    print("\nRound-trip verification...")
    reloaded = FragmentLibrary.load(pipeline.library_path)
    assert len(reloaded.records) == len(library.records), "Record count mismatch after reload"
    print("Round-trip OK.")


if __name__ == "__main__":
    main()
