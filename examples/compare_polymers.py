"""
Compare force-field parameters across two polymers and write a text report.

Shows the full study workflow: build → compare → save results → write report → plot.

Run:
    PATH="/Users/reinazheng/miniconda3/envs/md_engines/bin:$PATH" uv run python examples/compare_polymers.py
"""
from __future__ import annotations

from pathlib import Path

from polymer_md.analysis.comparator import ParameterComparator
from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.analysis.report import TextReport
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy

OUTPUT_DIR = Path("output/comparison_study")


def make_pipeline(seed: int) -> PolymerParameterisationPipeline:
    specs = [
        MonomerSpec(
            residue=MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S")),
            weight=1.0,
        ),
    ]
    return PolymerParameterisationPipeline(
        specs=specs,
        n=8,
        output_dir=OUTPUT_DIR / f"seed_{seed}",
        seed=seed,
        solver=ProportionalSolver(ht_fraction=0.95),
        conformer_generator=ETKDGConformerGenerator(use_uff=False, use_random_coords=True),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
        charge_method="bcc",
    )


fragments = [
    ExtractionSpec(
        pattern="[#6;A]-[#6;A]",
        parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
    ).to_fragment(),
    ExtractionSpec(
        pattern="[#6;A]",
        parameters=(AtomParameter.CHARGE,),
    ).to_fragment(),
]

molecules = {
    "seed_1": make_pipeline(seed=1).run(),
    "seed_2": make_pipeline(seed=2).run(),
}

comparison = ParameterComparator(fragments=fragments).compare(molecules)

comparison.save(OUTPUT_DIR / "results.json")
print(TextReport(comparison))

try:
    from polymer_md.visualisation.comparison import plot_comparison
    fig = plot_comparison(comparison)
    fig.savefig(OUTPUT_DIR / "comparison.pdf", bbox_inches="tight")
    print(f"\nPlot saved to {OUTPUT_DIR / 'comparison.pdf'}")
except ImportError:
    pass
