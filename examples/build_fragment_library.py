"""
Pre-build a fragment library for a monomer set and save it for later reuse.

Useful when you want to parameterise many polymers from the same monomers
without re-running acpype each time. Pass the saved library to a pipeline
via FragmentLibraryPipeline or load it directly with FragmentLibrary.load().

Run:
    PATH="/Users/reinazheng/miniconda3/envs/md_engines/bin:$PATH" uv run python examples/build_fragment_library.py
"""
from __future__ import annotations

from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragment_library_pipeline import FragmentLibraryPipeline
from polymer_md.parameterisation.fragments.library import FragmentLibrary

OUTPUT_DIR = Path("output/library_polystyrene")

specs = [
    MonomerSpec(
        residue=MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S")),
        weight=1.0,
    ),
]

pipeline = FragmentLibraryPipeline(
    specs=specs,
    output_dir=OUTPUT_DIR / "trimers",
    library_path=OUTPUT_DIR / "library.json",
    conformer_generator=ETKDGConformerGenerator(use_uff=False, use_random_coords=True),
    probability_threshold=0.0,
    charge_method="bcc",
)

library = pipeline.run()
print(f"Records:         {len(library.records)}")
print(f"Atom metadata:   {sum(len(m) for m in library.atom_metadata.values())} entries")
print(f"Saved to:        {pipeline.library_path}")

reloaded = FragmentLibrary.load(pipeline.library_path)
assert len(reloaded.records) == len(library.records)
print("Round-trip load: OK")
