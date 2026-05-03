"""
Integration tests for the fragment library and polymer parameterisation pipelines.

The real extraction, matching, library building, and tiling code runs without mocks.
The only patched boundary is `_parameterise_trimer`, which ordinarily calls OBabel and
acpype as subprocesses. Here it is replaced by a helper that builds a `ParameterisedTrimer`
from the real trimer mol (embedded with ETKDG) plus synthetic GAFF2-like parameters, so
the trimer structure is chemically meaningful without requiring external tools.

The real `TrimerParameterisationPipeline.run()` still executes: it builds all trimers,
applies the probability filter, and calls the (patched) `_parameterise_trimer` for each.
`FragmentLibraryBuilder.build()` and `PolymerParameterisationPipeline.run()` are entirely
un-mocked.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import parmed as pmd
import pytest
from rdkit import Chem

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.trimer import TrimerResult
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.conversion.gromacs_files import GromacsFiles
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter
from polymer_md.parameterisation.fragments.extraction.builder import FragmentLibraryBuilder
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.pipeline import TrimerParameterisationPipeline
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import ResiduePositionStrategy
from polymer_md.utils.parmed_helper import ParmedTypeResolver
from polymer_md.utils.topology_builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_specs() -> list[MonomerSpec]:
    residue = MonomerToResidueConverter.convert(Monomer(smiles="C=CC", label="P"))
    return [MonomerSpec(residue=residue, weight=1.0)]


def _add_synthetic_parameters(structure: pmd.Structure) -> None:
    for bond in structure.bonds:
        bond.type = pmd.BondType(k=310.0, req=1.526)
    for angle in structure.angles:
        angle.type = pmd.AngleType(k=50.0, theteq=109.5)
    for dihedral in structure.dihedrals:
        dihedral.type = pmd.DihedralType(phi_k=0.15, phase=0.0, per=3)
    for atom in structure.atoms:
        atom.charge = -0.06 if atom.atomic_number == 6 else 0.06
        atom.epsilon = 0.1094
        atom.sigma = 3.3997
        atom.mass = 12.011 if atom.atomic_number == 6 else 1.008
        atom.atom_type = pmd.AtomType(
            "c3" if atom.atomic_number == 6 else "hc",
            atom.idx,
            atom.mass,
            atom.atomic_number,
        )


def _make_fixture_parameterised_trimer(
    trimer: TrimerResult, output_dir: Path
) -> ParameterisedTrimer:
    """
    Constructs a ParameterisedTrimer from the real trimer mol with synthetic parameters,
    bypassing the OBabel/acpype subprocess calls.
    """
    mol_3d = ETKDGConformerGenerator().embed(trimer.mol)
    structure = TopologyBuilder.build(mol_3d)
    _add_synthetic_parameters(structure)
    dummy_gro = output_dir / "dummy.gro"
    dummy_top = output_dir / "dummy.top"
    dummy_itp = output_dir / "dummy.itp"
    for p in (dummy_gro, dummy_top, dummy_itp):
        p.touch()
    return ParameterisedTrimer(
        trimer_result=trimer,
        gromacs_files=GromacsFiles(itp=dummy_itp, gro=dummy_gro, top=dummy_top),
        structure=structure,
        mol_3d=mol_3d,
    )


def _build_trimers(specs: list[MonomerSpec], tmp: Path) -> list[ParameterisedTrimer]:
    with patch.object(
        TrimerParameterisationPipeline,
        "_parameterise_trimer",
        side_effect=_make_fixture_parameterised_trimer,
    ):
        pipeline = TrimerParameterisationPipeline(
            specs=specs,
            probability_threshold=0.0,
            conformer_generator=ETKDGConformerGenerator(),
        )
        return pipeline.run(tmp)


# ---------------------------------------------------------------------------
# Fragment library integration tests
# ---------------------------------------------------------------------------

class TestFragmentLibraryBuilderIntegration:
    def test_library_has_records_from_real_extraction(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmpdir:
            parameterised_trimers = _build_trimers(specs, Path(tmpdir) / "trimers")

        assert len(parameterised_trimers) > 0
        library = FragmentLibraryBuilder().build(parameterised_trimers)
        assert len(library.records) > 0

    def test_library_has_annotated_bonds(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmpdir:
            parameterised_trimers = _build_trimers(specs, Path(tmpdir) / "trimers")

        library = FragmentLibraryBuilder().build(parameterised_trimers)
        fragments = library.to_fragments()
        assert len(fragments) > 0
        assert any(len(f.annotated_bonds) > 0 for f in fragments)

    def test_library_has_atom_metadata(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmpdir:
            parameterised_trimers = _build_trimers(specs, Path(tmpdir) / "trimers")

        library = FragmentLibraryBuilder().build(parameterised_trimers)
        assert len(library.atom_metadata) > 0

    def test_library_round_trip_preserves_record_count(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            parameterised_trimers = _build_trimers(specs, tmp / "trimers")
            library = FragmentLibraryBuilder().build(parameterised_trimers)
            lib_path = tmp / "library.json"
            library.save(lib_path)
            reloaded = FragmentLibrary.load(lib_path)

        assert len(reloaded.records) == len(library.records)


# ---------------------------------------------------------------------------
# Polymer pipeline integration tests
# ---------------------------------------------------------------------------

class TestPolymerPipelineIntegration:
    @pytest.fixture
    def small_library(self, tmp_path) -> FragmentLibrary:
        specs = _make_specs()
        parameterised_trimers = _build_trimers(specs, tmp_path / "trimers")
        return FragmentLibraryBuilder().build(parameterised_trimers)

    def _run_pipeline(self, small_library: FragmentLibrary, tmp_path: Path) -> PolymerParameterisationPipeline:
        pipeline = PolymerParameterisationPipeline(
            specs=_make_specs(),
            n=4,
            output_dir=tmp_path / "polymer",
            seed=42,
            conformer_generator=ETKDGConformerGenerator(),
            missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)},
            adjust_charge=True,
        )
        with patch.object(
            PolymerParameterisationPipeline,
            "_build_library_for_polymer",
            return_value=small_library,
        ):
            return pipeline.run()

    def test_polymer_pipeline_produces_parameterised_molecule(self, small_library, tmp_path):
        result = self._run_pipeline(small_library, tmp_path)
        assert len(result.structure.atoms) > 0
        assert len(result.structure.bonds) > 0

    def test_polymer_pipeline_all_bonds_have_nonzero_types(self, small_library, tmp_path):
        result = self._run_pipeline(small_library, tmp_path)
        for bond in result.structure.bonds:
            bt = ParmedTypeResolver.bond_type(bond)
            assert bt.k > 0
            assert bt.req > 0

    def test_polymer_pipeline_charge_is_neutral(self, small_library, tmp_path):
        result = self._run_pipeline(small_library, tmp_path)
        total_charge = sum(a.charge for a in result.structure.atoms)
        assert abs(total_charge) < 1e-3

    def test_polymer_pipeline_output_files_written(self, small_library, tmp_path):
        result = self._run_pipeline(small_library, tmp_path)
        assert result.source.gro.exists()
        assert result.source.top.exists()
        assert result.source.itp.exists()
