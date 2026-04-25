from __future__ import annotations

import pytest
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.analysis.comparator import ParameterComparator
from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.analysis.results import ComparisonResult
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.utils.topology_builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Fixture: a simple propane structure with assigned bond and atom parameters
# ---------------------------------------------------------------------------

def _make_parameterised_propane() -> tuple[pmd.Structure, Chem.Mol]:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    structure = TopologyBuilder.build(mol)

    for atom in structure.atoms:
        atom.charge = -0.10
        atom.epsilon = 0.05
        atom.sigma = 0.35
        atom.mass = 12.0 if mol.GetAtomWithIdx(atom.idx).GetAtomicNum() == 6 else 1.008

    for bond in structure.bonds:
        bond.type = pmd.BondType(k=300.0, req=1.54)

    for angle in structure.angles:
        angle.type = pmd.AngleType(k=50.0, theteq=109.5)

    for dihedral in structure.dihedrals:
        dihedral.type = pmd.DihedralType(phi_k=0.5, phase=0.0, per=3)

    return structure, mol


@pytest.fixture
def parameterised_propane() -> tuple[pmd.Structure, Chem.Mol]:
    return _make_parameterised_propane()


class TestParameterComparatorExtract:
    def test_returns_entry_for_each_fragment(self, parameterised_propane):
        structure, mol = parameterised_propane
        spec = ExtractionSpec(pattern="[#6;A]-[#6;A]", parameters=(AtomParameter.CHARGE,))
        fragment = spec.to_fragment()
        comparator = ParameterComparator(fragments=[fragment])

        result = comparator.extract(structure)

        assert fragment in result

    def test_extracts_atom_charges(self, parameterised_propane):
        structure, mol = parameterised_propane
        spec = ExtractionSpec(pattern="[#6;A]", parameters=(AtomParameter.CHARGE,))
        fragment = spec.to_fragment()
        comparator = ParameterComparator(fragments=[fragment])

        result = comparator.extract(structure)
        values = result.get(fragment, {}).get(AtomParameter.CHARGE, [])

        assert len(values) > 0
        assert all(pytest.approx(v, abs=1e-6) == -0.10 for v in values)

    def test_extracts_bond_force_constants(self, parameterised_propane):
        structure, mol = parameterised_propane
        spec = ExtractionSpec(pattern="[#6;A]-[#6;A]", parameters=(BondParameter.FORCE_CONSTANT,))
        fragment = spec.to_fragment()
        comparator = ParameterComparator(fragments=[fragment])

        result = comparator.extract(structure)
        values = result.get(fragment, {}).get(BondParameter.FORCE_CONSTANT, [])

        assert len(values) > 0
        assert all(pytest.approx(v, abs=1e-6) == 300.0 for v in values)

    def test_invalid_pattern_returns_empty_for_fragment(self, parameterised_propane):
        structure, _ = parameterised_propane
        from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
        fragment = Fragment(pattern="[#99]")
        comparator = ParameterComparator(fragments=[fragment])

        result = comparator.extract(structure)

        assert result[fragment] == {}

    def test_multiple_fragments_all_present(self, parameterised_propane):
        structure, _ = parameterised_propane
        spec1 = ExtractionSpec(pattern="[#6;A]", parameters=(AtomParameter.CHARGE,))
        spec2 = ExtractionSpec(pattern="[#6;A]-[#6;A]", parameters=(BondParameter.FORCE_CONSTANT,))
        f1 = spec1.to_fragment()
        f2 = spec2.to_fragment()
        comparator = ParameterComparator(fragments=[f1, f2])

        result = comparator.extract(structure)

        assert f1 in result
        assert f2 in result

    def test_invalid_smarts_pattern_returns_empty_dict(self, parameterised_propane):
        structure, _ = parameterised_propane
        fragment = Fragment(pattern="[invalid!!!")
        comparator = ParameterComparator(fragments=[fragment])

        result = comparator.extract(structure)

        assert result[fragment] == {}

    def test_untyped_bond_silently_skipped_in_extraction(self):
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        structure = TopologyBuilder.build(mol)

        spec = ExtractionSpec(pattern="[#6]-[#6]", parameters=(BondParameter.FORCE_CONSTANT,))
        fragment = spec.to_fragment()
        comparator = ParameterComparator(fragments=[fragment])

        result = comparator.extract(structure)

        assert BondParameter.FORCE_CONSTANT not in result.get(fragment, {})


class TestParameterComparatorCompare:
    def test_compare_returns_comparison_result(self, parameterised_propane):
        structure, mol = parameterised_propane
        spec = ExtractionSpec(pattern="[#6;A]", parameters=(AtomParameter.CHARGE,))
        fragment = spec.to_fragment()
        comparator = ParameterComparator(fragments=[fragment])

        from polymer_md.conversion.gromacs_files import GromacsFiles
        import tempfile
        from pathlib import Path

        # Create dummy GROMACS files to satisfy ParameterisedMolecule
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for name in ["mol.itp", "mol.gro", "mol.top"]:
                (tmp / name).write_text("")
            # GromacsFiles asserts existence so files must be there
            gfiles = GromacsFiles(itp=tmp / "mol.itp", gro=tmp / "mol.gro", top=tmp / "mol.top")
            pm = ParameterisedMolecule(structure=structure, mol=mol, source=gfiles)
            result = comparator.compare({"propane": pm})

        assert isinstance(result, ComparisonResult)
        assert "propane" in result.results

    def test_compare_summarise_gives_statistics(self, parameterised_propane):
        structure, mol = parameterised_propane
        spec = ExtractionSpec(pattern="[#6;A]", parameters=(AtomParameter.CHARGE,))
        fragment = spec.to_fragment()
        comparator = ParameterComparator(fragments=[fragment])

        from polymer_md.conversion.gromacs_files import GromacsFiles
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for name in ["mol.itp", "mol.gro", "mol.top"]:
                (tmp / name).write_text("")
            gfiles = GromacsFiles(itp=tmp / "mol.itp", gro=tmp / "mol.gro", top=tmp / "mol.top")
            pm = ParameterisedMolecule(structure=structure, mol=mol, source=gfiles)
            comparison = comparator.compare({"propane": pm})

        summary = comparison.summarise()
        assert "propane" in summary
        assert fragment in summary["propane"]
        assert AtomParameter.CHARGE in summary["propane"][fragment]
        s = summary["propane"][fragment][AtomParameter.CHARGE]
        assert s.count > 0
        assert pytest.approx(s.mean, abs=1e-6) == -0.10
