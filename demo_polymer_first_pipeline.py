"""
Demo: Polymer-first parameterisation pipeline for MMA/BA/MAA copolymer.

Instead of pre-building a library from all possible trimers above a probability
threshold, this approach:
  1. Builds the polymer sequence first (no parameterisation yet).
  2. Extracts the unique (left, centre, right) trimer environments that actually
     appear in that specific polymer.
  3. Parameterises only those trimers.
  4. Builds the fragment library from those trimers.
  5. Tiles parameters onto the polymer.

This guarantees complete coverage for the polymer at hand and avoids
parameterising trimers that never appear.

Requires: OBabel and acpype installed on PATH.

Usage:
    python demo_polymer_first_pipeline.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import parmed as pmd
from rdkit import Chem

from polymer_md.analysis.comparator import ParameterComparator
from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AtomParameter,
    BondParameter,
)
from polymer_md.parameterisation.polymer_first_pipeline import (
    PolymerFirstParameterisationPipeline,
)
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

REFERENCE_DIR = Path("tests/full_polymer_results/mmabamaa20")
OUTPUT_DIR = Path("output/mmabamaa_polymer_first")


def _make_specs() -> list[MonomerSpec]:
    mma = MonomerToResidueConverter.convert(Monomer(smiles="C=C(C)C(=O)OC", label="MMA"))
    ba = MonomerToResidueConverter.convert(Monomer(smiles="C=CC(=O)OCCCC", label="BA"))
    maa = MonomerToResidueConverter.convert(Monomer(smiles="C=C(C)C(=O)O", label="MAA"))
    return [
        MonomerSpec(residue=mma, weight=0.60),
        MonomerSpec(residue=ba, weight=0.20),
        MonomerSpec(residue=maa, weight=0.20),
    ]


def _mol_from_structure(structure: pmd.Structure) -> Chem.Mol:
    from rdkit.Chem import Atom, RWMol

    rw = RWMol()
    for atom in structure.atoms:
        rw.AddAtom(Atom(atom.atomic_number))
    for bond in structure.bonds:
        a1, a2 = bond.atom1, bond.atom2
        a1_type = getattr(a1.atom_type, "name", "") if a1.atom_type else ""
        a2_type = getattr(a2.atom_type, "name", "") if a2.atom_type else ""
        is_carbonyl = ("c" == a1_type and a2_type == "o") or (
            "c" == a2_type and a1_type == "o"
        )
        bond_type = Chem.BondType.DOUBLE if is_carbonyl else Chem.BondType.SINGLE
        rw.AddBond(a1.idx, a2.idx, bond_type)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _load_reference() -> ParameterisedMolecule:
    itp_path = REFERENCE_DIR / "MMABAMAA20_GMX.itp"
    structure = pmd.load_file(str(itp_path))
    mol = _mol_from_structure(structure)
    return ParameterisedMolecule(structure=structure, mol=mol, source=None)


def _run_comparison(
    our_mol: ParameterisedMolecule,
    reference_mol: ParameterisedMolecule,
) -> None:
    print("\n=== Parameter Comparison: Polymer-First Pipeline vs. Reference ===\n")

    specs = [
        ExtractionSpec(
            pattern="[#6;A]-[#6;A]",
            parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
        ),
        ExtractionSpec(
            pattern="[#6;A]-[#8;A]",
            parameters=(BondParameter.FORCE_CONSTANT, BondParameter.EQUILIBRIUM_LENGTH),
        ),
        ExtractionSpec(
            pattern="[#6;A]",
            parameters=(AtomParameter.CHARGE,),
        ),
    ]
    fragments = [s.to_fragment() for s in specs]

    comparator = ParameterComparator(fragments=fragments)
    comparison = comparator.compare(
        {"reference": reference_mol, "polymer_first": our_mol}
    )
    summary = comparison.summarise()

    for label, frag_map in summary.items():
        print(f"--- {label} ---")
        for frag, param_map in frag_map.items():
            for param, s in param_map.items():
                print(
                    f"  {frag.pattern:20s} {param.name:25s}"
                    f"  mean={s.mean:8.4f}  std={s.std:7.4f}  n={s.count}"
                )
        print()


def main() -> None:
    print("=== MMA / BA / MAA Copolymer — Polymer-First Pipeline Demo ===\n")

    specs = _make_specs()

    pipeline = PolymerFirstParameterisationPipeline(
        specs=specs,
        n=20,
        output_dir=OUTPUT_DIR,
        seed=42,
        solver=ProportionalSolver(ht_fraction=1.0),
        conformer_generator=ETKDGConformerGenerator(
            use_uff=False, use_random_coords=True
        ),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
        charge_method="bcc",
    )

    print("Running polymer-first pipeline (builds polymer, extracts needed trimers, tiles)...")
    our_mol, library = pipeline.run()

    total_charge = sum(a.charge for a in our_mol.structure.atoms)
    print(
        f"\nPolymer-first result: {len(our_mol.structure.atoms)} atoms, "
        f"charge={total_charge:.4f}"
    )
    print(f"  Library: {len(library.records)} records")
    print(f"  GRO: {our_mol.source.gro}")
    print(f"  TOP: {our_mol.source.top}")

    print("\nLoading reference MMABAMAA20 structure...")
    reference_mol = _load_reference()
    print(f"Reference: {len(reference_mol.structure.atoms)} atoms")

    _run_comparison(our_mol, reference_mol)


if __name__ == "__main__":
    main()
