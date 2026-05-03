"""
Demo: ParameterComparator

Compares bond force constants extracted from two polystyrene 8-mers built with
different random seeds. Each polymer's fragment library is built on-demand from
the trimers required by that polymer's actual sequence.

Requires: OBabel and acpype installed on PATH.

Usage:
    conda run -n md_engines python demo_analysis.py
"""
import logging
from pathlib import Path

from polymer_md.analysis.comparator import ParameterComparator
from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")


def build_molecule(seed: int, output_dir: Path):
    styrene_residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    specs = [MonomerSpec(residue=styrene_residue, weight=1.0)]
    pipeline = PolymerParameterisationPipeline(
        specs=specs,
        n=8,
        output_dir=output_dir,
        seed=seed,
        conformer_generator=ETKDGConformerGenerator(use_uff=False, use_random_coords=True),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
        charge_method="bcc",
    )
    return pipeline.run()


def main() -> None:
    print("=== Analysis / ParameterComparator Demo ===\n")

    print("Building two 8-mer polystyrene polymers (seeds 1 and 2)...")
    mol_a = build_molecule(seed=1, output_dir=Path("output/demo_analysis/seed1"))
    mol_b = build_molecule(seed=2, output_dir=Path("output/demo_analysis/seed2"))

    print("\nExtracting aliphatic C-C bond parameters for comparison...")
    spec = ExtractionSpec(
        pattern="[#6;A]-[#6;A]",
        parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
    )
    fragment = spec.to_fragment()

    comparator = ParameterComparator(fragments=[fragment])
    comparison = comparator.compare({"seed_1": mol_a, "seed_2": mol_b})
    summary = comparison.summarise()

    for label, frag_map in summary.items():
        print(f"\n--- {label} ---")
        for frag, param_map in frag_map.items():
            print(f"  Fragment: {frag.pattern}")
            for param, s in param_map.items():
                print(f"    {param.name}: mean={s.mean:.3f}, std={s.std:.3f}, n={s.count}")


if __name__ == "__main__":
    main()
