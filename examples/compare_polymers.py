"""
Compare our MMA/BA/MAA polymer parameters against the reference MMABAMAA20 force field,
then render a 3D difference view with atoms coloured by % deviation from reference.

Run:
    PATH="/Users/reinazheng/miniconda3/envs/md_engines/bin:$PATH" uv run python examples/compare_polymers.py
"""

from __future__ import annotations

from pathlib import Path

import parmed as pmd

from polymer_md.analysis.comparator import ParameterComparator
from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.analysis.report import TextReport
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.conversion.gromacs_files import GromacsFiles
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.data_models.parameterised_mol import (
    ParameterisedMolecule,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AtomParameter,
    BondParameter,
)
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import (
    ResiduePositionStrategy,
)
from polymer_md.utils.parmed_helper import mol_from_structure
from polymer_md.visualisation.difference_3d import DifferenceViewer

OUTPUT_DIR = Path("output/comparison_mmabamaa_vs_reference")
REFERENCE_DIR = Path("tests/full_polymer_results/mmabamaa20_dpnb_4wt")
REFERENCE_ITP = REFERENCE_DIR / "MMABAMAA20_GMX.itp"
TRIMER_CACHE_DIR = Path("output/mmabamaa_polymer/trimers")


def build_pipeline() -> PolymerParameterisationPipeline:
    specs = [
        MonomerSpec(
            residue=MonomerToResidueConverter.convert(
                Monomer(smiles="C=C(C)C(=O)OC", label="MMA")
            ),
            weight=0.5,
        ),
        MonomerSpec(
            residue=MonomerToResidueConverter.convert(
                Monomer(smiles="C=CC(=O)OCCCC", label="BA")
            ),
            weight=0.3,
        ),
        MonomerSpec(
            residue=MonomerToResidueConverter.convert(
                Monomer(smiles="C=C(C)C(=O)O", label="MAA")
            ),
            weight=0.2,
        ),
    ]
    return PolymerParameterisationPipeline(
        specs=specs,
        n=20,
        output_dir=OUTPUT_DIR / "pipeline",
        seed=42,
        solver=ProportionalSolver(ht_fraction=1.0),
        conformer_generator=ETKDGConformerGenerator(
            use_uff=False, use_random_coords=True
        ),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
        charge_method="bcc",
        trimer_cache_dir=TRIMER_CACHE_DIR if TRIMER_CACHE_DIR.exists() else None,
    )


def load_reference() -> ParameterisedMolecule:
    source = GromacsFiles(
        itp=REFERENCE_ITP,
        gro=REFERENCE_DIR / "mmabamaa20_dpnb4_init.gro",
        top=REFERENCE_DIR / "topol.top",
    )
    structure = pmd.load_file(str(REFERENCE_ITP))
    mol = mol_from_structure(structure)
    return ParameterisedMolecule(structure=structure, mol=mol, source=source)


fragments = [
    ExtractionSpec(
        pattern="[#6;A]-[#6;A]",
        parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
    ).to_fragment(),
    ExtractionSpec(
        pattern="[#6;A]-[#8;A]",
        parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
    ).to_fragment(),
    ExtractionSpec(
        pattern="[#6;A]",
        parameters=(AtomParameter.CHARGE,),
    ).to_fragment(),
]

our_polymer = build_pipeline().run()
reference = load_reference()

molecules = {
    "our_pipeline": our_polymer,
    "reference": reference,
}

comparison = ParameterComparator(fragments=fragments).compare(molecules)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
comparison.save(OUTPUT_DIR / "results.json")
print(TextReport(comparison))

try:
    from polymer_md.visualisation.comparison import plot_comparison

    fig = plot_comparison(comparison)
    fig.savefig(OUTPUT_DIR / "comparison.pdf", bbox_inches="tight")
    print(f"\nPlot saved to {OUTPUT_DIR / 'comparison.pdf'}")
except Exception:
    pass

viewer = DifferenceViewer(
    molecule=our_polymer,
    reference=reference,
    fragments=fragments,
)
diff_path = viewer.save(OUTPUT_DIR / "difference_3d.html")
print(f"3D difference view saved to {diff_path}")
