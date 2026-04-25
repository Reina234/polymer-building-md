from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import parmed as pmd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.trimer import BondType, TrimerResult
from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.pipeline import TrimerParameterisationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_specs() -> list[MonomerSpec]:
    from polymer_md.building.monomer_converter import MonomerToResidueConverter
    from polymer_md.core.monomer import Monomer
    residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    return [MonomerSpec(residue=residue, weight=1.0)]


def _make_minimal_structure() -> pmd.Structure:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    from polymer_md.utils.topology_builder import TopologyBuilder
    return TopologyBuilder.build(mol)


# ---------------------------------------------------------------------------
# Tests: internal helpers (no external tools needed)
# ---------------------------------------------------------------------------

class TestTrimerPipelineHelpers:
    def test_build_residue_map_keys_match_spec_ids(self):
        specs = _make_specs()
        pipeline = TrimerParameterisationPipeline(specs=specs)
        residues = pipeline._build_residue_map()
        assert set(residues.keys()) == {spec.residue_id for spec in specs}

    def test_solve_transition_matrix_returns_matrix(self):
        specs = _make_specs()
        pipeline = TrimerParameterisationPipeline(specs=specs)
        matrix = pipeline._solve_transition_matrix()
        assert matrix is not None
        assert matrix.n_sites > 0

    def test_filter_by_probability_removes_below_threshold(self):
        specs = _make_specs()
        pipeline = TrimerParameterisationPipeline(specs=specs, probability_threshold=0.5)
        mock_trimers = [
            MagicMock(probability=0.6),
            MagicMock(probability=0.3),
            MagicMock(probability=0.8),
        ]
        result = pipeline._filter_by_probability(mock_trimers)
        assert len(result) == 2
        assert all(t.probability >= 0.5 for t in result)

    def test_filter_by_probability_keeps_all_when_threshold_zero(self):
        specs = _make_specs()
        pipeline = TrimerParameterisationPipeline(specs=specs, probability_threshold=0.0)
        mock_trimers = [MagicMock(probability=0.01), MagicMock(probability=0.001)]
        result = pipeline._filter_by_probability(mock_trimers)
        assert len(result) == 2

    def test_trimer_name_is_deterministic(self):
        from polymer_md.building.trimer_builder import TrimerBuilder
        specs = _make_specs()
        pipeline = TrimerParameterisationPipeline(specs=specs)
        residues = pipeline._build_residue_map()
        matrix = pipeline._solve_transition_matrix()
        trimers = TrimerBuilder(residues=residues, matrix=matrix, cap=pipeline.cap).build_all()
        assert trimers, "Expected at least one trimer"
        name1 = TrimerParameterisationPipeline._trimer_name(trimers[0])
        name2 = TrimerParameterisationPipeline._trimer_name(trimers[0])
        assert name1 == name2

    def test_trimer_name_contains_residue_ids(self):
        from polymer_md.building.trimer_builder import TrimerBuilder
        specs = _make_specs()
        pipeline = TrimerParameterisationPipeline(specs=specs)
        residues = pipeline._build_residue_map()
        matrix = pipeline._solve_transition_matrix()
        trimers = TrimerBuilder(residues=residues, matrix=matrix, cap=pipeline.cap).build_all()
        name = TrimerParameterisationPipeline._trimer_name(trimers[0])
        # Must contain residue IDs
        assert "S" in name


# ---------------------------------------------------------------------------
# Tests: run (fully mocked external tools)
# ---------------------------------------------------------------------------

class TestTrimerPipelineRun:
    def test_run_returns_list_of_parameterised_trimers(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch.object(TrimerParameterisationPipeline, "_parameterise_trimer") as mock_param,
            ):
                from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
                mock_param.return_value = MagicMock(spec=ParameterisedTrimer)

                pipeline = TrimerParameterisationPipeline(
                    specs=specs,
                    probability_threshold=0.0,
                )
                results = pipeline.run(tmp_path)

        assert isinstance(results, list)
        assert len(results) > 0
        assert mock_param.called

    def test_run_calls_parameterise_trimer_for_each_selected(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.object(TrimerParameterisationPipeline, "_parameterise_trimer") as mock_param:
                mock_param.return_value = MagicMock()
                pipeline = TrimerParameterisationPipeline(specs=specs, probability_threshold=0.0)
                results = pipeline.run(tmp_path)

        # One call per selected trimer
        assert mock_param.call_count == len(results)

    def test_run_probability_filter_applied(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.object(TrimerParameterisationPipeline, "_parameterise_trimer") as mock_param:
                mock_param.return_value = MagicMock()
                # High threshold → fewer trimers selected
                pipeline_high = TrimerParameterisationPipeline(
                    specs=specs, probability_threshold=0.99
                )
                results_high = pipeline_high.run(tmp_path)

                mock_param.reset_mock()
                pipeline_low = TrimerParameterisationPipeline(
                    specs=specs, probability_threshold=0.0
                )
                results_low = pipeline_low.run(tmp_path)

        # Low threshold keeps all trimers, high might drop some
        assert len(results_low) >= len(results_high)

    def test_run_creates_output_directory(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pipeline_out" / "nested"
            with patch.object(TrimerParameterisationPipeline, "_parameterise_trimer") as mock_param:
                mock_param.return_value = MagicMock()
                pipeline = TrimerParameterisationPipeline(specs=specs, probability_threshold=0.99)
                pipeline.run(output)
            assert output.exists()

    def test_parameterise_trimer_called_with_trimer_and_dir(self):
        specs = _make_specs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.object(TrimerParameterisationPipeline, "_parameterise_trimer") as mock_param:
                mock_param.return_value = MagicMock()
                pipeline = TrimerParameterisationPipeline(specs=specs, probability_threshold=0.0)
                pipeline.run(tmp_path)

        for call in mock_param.call_args_list:
            trimer_arg, dir_arg = call.args
            assert dir_arg == tmp_path


# ---------------------------------------------------------------------------
# Tests: _parameterise_trimer (mocked external tools)
# ---------------------------------------------------------------------------

def _make_trimer_result() -> TrimerResult:
    mol = Chem.MolFromSmiles("CCCCCC")
    return TrimerResult(
        left_id="S",
        central_id="S",
        right_id="S",
        left_bond=BondType(from_site=0, to_site=1),
        right_bond=BondType(from_site=1, to_site=0),
        mol=mol,
        probability=1.0,
        left_atom_indices=frozenset({0, 1}),
        central_atom_indices=frozenset({2, 3}),
        right_atom_indices=frozenset({4, 5}),
        cap_atom_indices=frozenset(),
    )


class TestParameteriseTrimerMethod:
    def _run_parameterise(self, trimer: TrimerResult, tmp: Path):
        mock_mol_3d = MagicMock()
        mock_conformer = MagicMock()
        mock_conformer.embed.return_value = mock_mol_3d
        mock_gromacs = MagicMock()
        mock_structure = MagicMock()

        with (
            patch("polymer_md.parameterisation.pipeline.RDKitHelper") as MockRDKit,
            patch("polymer_md.parameterisation.pipeline.OBabelConverter") as MockOBabel,
            patch("polymer_md.parameterisation.pipeline.AcpypeConverter") as MockAcpype,
            patch("polymer_md.parameterisation.pipeline.pmd.load_file", return_value=mock_structure),
        ):
            MockRDKit.write_sdf.return_value = None
            MockRDKit.canonical_smiles_stripped.return_value = "CCCCCC"
            MockOBabel.return_value.convert.return_value = tmp / "mol.mol2"
            MockAcpype.return_value.convert.return_value = mock_gromacs

            pipeline = TrimerParameterisationPipeline(
                specs=_make_specs(),
                conformer_generator=mock_conformer,
            )
            result = pipeline._parameterise_trimer(trimer, tmp)
            return result, mock_mol_3d, mock_conformer, mock_gromacs, mock_structure

    def test_returns_parameterised_trimer_type(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            result, *_ = self._run_parameterise(trimer, Path(tmp))
        assert isinstance(result, ParameterisedTrimer)

    def test_trimer_result_is_preserved(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            result, *_ = self._run_parameterise(trimer, Path(tmp))
        assert result.trimer_result is trimer

    def test_structure_is_from_pmd_load_file(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, _, mock_structure = self._run_parameterise(trimer, Path(tmp))
        assert result.structure is mock_structure

    def test_mol_3d_is_from_conformer_generator(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            result, mock_mol_3d, *_ = self._run_parameterise(trimer, Path(tmp))
        assert result.mol_3d is mock_mol_3d

    def test_gromacs_files_is_from_acpype(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, mock_gromacs, _ = self._run_parameterise(trimer, Path(tmp))
        assert result.gromacs_files is mock_gromacs

    def test_conformer_generator_embed_called_with_trimer_mol(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            _, _, mock_conformer, _, _ = self._run_parameterise(trimer, Path(tmp))
        mock_conformer.embed.assert_called_once_with(trimer.mol)

    def test_output_subdirectory_is_created(self):
        trimer = _make_trimer_result()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._run_parameterise(trimer, tmp_path)
            subdirs = list(tmp_path.iterdir())
            assert len(subdirs) == 1
            assert subdirs[0].is_dir()
