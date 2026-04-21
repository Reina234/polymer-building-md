import logging
from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.caps import BuiltinCap
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.pipeline import TrimerParameterisationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)


def main() -> None:
    print("=== polymer-md: trimer parameterisation pipeline ===\n")

    print("Converting monomers...")
    styrene_residue = MonomerToResidueConverter.convert(
        Monomer(smiles="C=Cc1ccccc1", label="S")
    )
    propylene_residue = MonomerToResidueConverter.convert(
        Monomer(smiles="CC=C", label="P")
    )
    print(f"  styrene  residue SMILES: {styrene_residue.residue_smiles}")
    print(f"  propylene residue SMILES: {propylene_residue.residue_smiles}")

    specs = [
        MonomerSpec(residue=styrene_residue, weight=0.7),
        MonomerSpec(residue=propylene_residue, weight=0.3),
    ]

    pipeline = TrimerParameterisationPipeline(
        specs=specs,
        cap=BuiltinCap.METHYL,
        conformer_generator=ETKDGConformerGenerator(use_uff=False),
        charge_method="bcc",
    )

    output_dir = Path("output/trimers")
    print(f"\nRunning pipeline → output: {output_dir}\n")
    parameterised_trimers = pipeline.run(output_dir=output_dir)

    print(f"\n=== Done: {len(parameterised_trimers)} parameterised trimers ===\n")
    for pt in parameterised_trimers:
        tr = pt.trimer_result
        print(
            f"  {tr.left_id}-{tr.central_id}-{tr.right_id}"
            f"  orientation={tr.orientation.name}"
            f"  p={tr.probability:.4f}"
        )
        print(f"    central atoms : {sorted(tr.central_atom_indices)}")
        print(f"    left atoms    : {sorted(tr.left_atom_indices)}")
        print(f"    right atoms   : {sorted(tr.right_atom_indices)}")
        print(f"    cap atoms     : {sorted(tr.cap_atom_indices)}")
        print(f"    GRO           : {pt.gromacs_files.gro}")
        print(f"    ITP           : {pt.gromacs_files.itp}")
        print()


if __name__ == "__main__":
    main()
