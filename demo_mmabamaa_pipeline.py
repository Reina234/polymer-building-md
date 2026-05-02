"""
Demo: end-to-end pipeline for MMA/BA/MAA copolymer, with comparison against a
reference GAFF2 parameterisation (tests/full_polymer_results/mmabamaa20).

Runs:
  1. Fragment library pipeline for MMA, BA, and MAA monomers.
  2. Polymer parameterisation pipeline for a 20-mer with matching composition.
  3. ParameterComparator between our output and the reference structure.

Requires: OBabel and acpype installed on PATH.

Usage:
    python demo_mmabamaa_pipeline.py
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
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.data_models.parameterised_mol import (
    ParameterisedMolecule,
)
from polymer_md.parameterisation.fragment_library_pipeline import (
    FragmentLibraryPipeline,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AtomParameter,
    BondParameter,
)
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import (
    ResiduePositionStrategy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

REFERENCE_DIR = Path("tests/full_polymer_results/mmabamaa20")
LIBRARY_PATH = Path("output/mmabamaa_library/library.json")


def _make_specs() -> list[MonomerSpec]:
    mma = MonomerToResidueConverter.convert(
        Monomer(smiles="C=C(C)C(=O)OC", label="MMA")
    )
    ba = MonomerToResidueConverter.convert(Monomer(smiles="C=CC(=O)OCCCC", label="BA"))
    maa = MonomerToResidueConverter.convert(Monomer(smiles="C=C(C)C(=O)O", label="MAA"))
    return [
        MonomerSpec(residue=mma, weight=0.60, ht_fraction=1),
        MonomerSpec(residue=ba, weight=0.20, ht_fraction=1),
        MonomerSpec(residue=maa, weight=0.20, ht_fraction=1),
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


def _build_library(specs: list[MonomerSpec]) -> FragmentLibrary:
    if LIBRARY_PATH.exists():
        print(f"Loading existing library from {LIBRARY_PATH}...")
        return FragmentLibrary.load(LIBRARY_PATH)

    print("Building fragment library for MMA / BA / MAA monomers...")
    pipeline = FragmentLibraryPipeline(
        specs=specs,
        output_dir=Path("output/mmabamaa_library/trimers"),
        library_path=LIBRARY_PATH,
        conformer_generator=ETKDGConformerGenerator(use_uff=False),
        probability_threshold=0.003,
        charge_method="bcc",
    )
    return pipeline.run()


def _build_polymer(
    library: FragmentLibrary, specs: list[MonomerSpec]
) -> ParameterisedMolecule:
    pipeline = PolymerParameterisationPipeline(
        library=library,
        specs=specs,
        n=20,
        output_dir=Path("output/mmabamaa_polymer"),
        seed=42,
        conformer_generator=ETKDGConformerGenerator(
            use_uff=False, use_random_coords=True
        ),
        missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
        adjust_charge=True,
    )
    print("Building and parameterising 20-mer copolymer...")
    return pipeline.run()


def _run_comparison(
    our_mol: ParameterisedMolecule,
    reference_mol: ParameterisedMolecule,
) -> None:
    print("\n=== Parameter Comparison: Our Pipeline vs. Reference ===\n")

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
        {"reference": reference_mol, "our_pipeline": our_mol}
    )
    summary = comparison.summarise()

    for label, frag_map in summary.items():
        print(f"--- {label} ---")
        for frag, param_map in frag_map.items():
            for param, s in param_map.items():
                print(
                    f"  {frag.pattern:20s} {param.name:25s}  mean={s.mean:8.4f}  std={s.std:7.4f}  n={s.count}"
                )
        print()


def main() -> None:
    print("=== MMA / BA / MAA Copolymer Pipeline Demo ===\n")

    specs = _make_specs()

    library = _build_library(specs)
    print(f"Library ready: {len(library.records)} records\n")

    our_mol = _build_polymer(library, specs)
    total_charge = sum(a.charge for a in our_mol.structure.atoms)
    print(
        f"\nOur polymer: {len(our_mol.structure.atoms)} atoms, charge={total_charge:.4f}"
    )
    print(f"  GRO: {our_mol.source.gro}")
    print(f"  TOP: {our_mol.source.top}")

    print("\nLoading reference MMABAMAA20 structure...")
    reference_mol = _load_reference()
    print(f"Reference: {len(reference_mol.structure.atoms)} atoms")

    _run_comparison(our_mol, reference_mol)


if __name__ == "__main__":
    main()
