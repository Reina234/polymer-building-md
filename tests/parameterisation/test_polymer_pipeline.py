from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import parmed as pmd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.core.residue_instance import ResidueInstance, ResidueType
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline


def _specs() -> list[MonomerSpec]:
    from polymer_md.building.monomer_converter import MonomerToResidueConverter
    from polymer_md.core.monomer import Monomer
    residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    return [MonomerSpec(residue=residue, weight=1.0)]


def _empty_library() -> FragmentLibrary:
    return FragmentLibrary(records=(), atom_metadata={})


class TestBuildPolymerAtomMetadata:
    def test_excludes_cap_residues(self):
        from polymer_md.core.polymer import Polymer
        from polymer_md.core.residue_instance import RESIDUE_TAG
        from rdkit.Chem import RWMol

        rw = RWMol(Chem.MolFromSmiles("CCC"))
        for i, atom in enumerate(rw.GetAtoms()):
            atom.SetIntProp(RESIDUE_TAG, i + 1)
        mol = rw.GetMol()

        monomer_instance = ResidueInstance(
            residue_id="S",
            instance_number=0,
            residue_tags=frozenset({1, 2}),
            residue_type=ResidueType.MONOMER,
        )
        cap_instance = ResidueInstance(
            residue_id="Me",
            instance_number=0,
            residue_tags=frozenset({3}),
            residue_type=ResidueType.CAP,
        )
        polymer = Polymer(mol=mol, residue_instances=[monomer_instance, cap_instance])

        metadata = PolymerParameterisationPipeline._build_polymer_atom_metadata(polymer)

        cap_idx = next(
            a.GetIdx() for a in mol.GetAtoms()
            if a.HasProp(RESIDUE_TAG) and a.GetIntProp(RESIDUE_TAG) == 3
        )
        assert cap_idx not in metadata

    def test_position_is_zero_indexed_rank(self):
        from polymer_md.core.polymer import Polymer
        from polymer_md.core.residue_instance import RESIDUE_TAG
        from rdkit.Chem import RWMol

        rw = RWMol(Chem.MolFromSmiles("CCC"))
        for i, atom in enumerate(rw.GetAtoms()):
            atom.SetIntProp(RESIDUE_TAG, i + 1)
        mol = rw.GetMol()

        instance = ResidueInstance(
            residue_id="S",
            instance_number=0,
            residue_tags=frozenset({1, 2, 3}),
            residue_type=ResidueType.MONOMER,
        )
        polymer = Polymer(mol=mol, residue_instances=[instance])

        metadata = PolymerParameterisationPipeline._build_polymer_atom_metadata(polymer)

        heavy_indices = sorted(
            a.GetIdx() for a in mol.GetAtoms()
            if a.HasProp(RESIDUE_TAG) and a.GetIntProp(RESIDUE_TAG) in {1, 2, 3}
            and a.GetAtomicNum() != 1
        )
        for expected_pos, idx in enumerate(heavy_indices):
            assert metadata[idx] == ("S", expected_pos)

    def test_residue_id_preserved(self):
        from polymer_md.core.polymer import Polymer
        from polymer_md.core.residue_instance import RESIDUE_TAG
        from rdkit.Chem import RWMol

        rw = RWMol(Chem.MolFromSmiles("C"))
        rw.GetAtomWithIdx(0).SetIntProp(RESIDUE_TAG, 1)
        mol = rw.GetMol()

        instance = ResidueInstance(
            residue_id="MYRESID",
            instance_number=0,
            residue_tags=frozenset({1}),
            residue_type=ResidueType.MONOMER,
        )
        polymer = Polymer(mol=mol, residue_instances=[instance])

        metadata = PolymerParameterisationPipeline._build_polymer_atom_metadata(polymer)

        for residue_id, _ in metadata.values():
            assert residue_id == "MYRESID"


class TestPolymerPipelineRun:
    def test_run_produces_parameterised_molecule(self):
        from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Build a minimal mol for mocking
            mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

            mock_polymer = MagicMock()
            mock_polymer.mol = Chem.MolFromSmiles("CCC")
            mock_polymer.residue_instances = []

            with (
                patch.object(PolymerParameterisationPipeline, "_build_polymer", return_value=mock_polymer),
                patch("polymer_md.parameterisation.polymer_pipeline.ETKDGConformerGenerator") as MockConformer,
                patch("polymer_md.parameterisation.polymer_pipeline.TopologyBuilder") as MockTB,
                patch("polymer_md.parameterisation.polymer_pipeline.StructureMolDeriver") as MockDeriv,
                patch("polymer_md.parameterisation.polymer_pipeline.PolymerParameterisationTiler") as MockTiler,
                patch("polymer_md.parameterisation.polymer_pipeline.adjust_charge_neutrality"),
            ):
                mock_structure = MagicMock(spec=pmd.Structure)
                mock_atoms = [MagicMock() for _ in range(3)]
                mock_structure.atoms = mock_atoms
                MockTB.build.return_value = mock_structure
                MockDeriv.derive.return_value = mol
                MockConformer.return_value.embed.return_value = mol
                mock_tiler_instance = MagicMock()
                MockTiler.return_value = mock_tiler_instance

                # Patch _save_gromacs to avoid actual file I/O
                with patch.object(PolymerParameterisationPipeline, "_save_gromacs") as mock_save:
                    from polymer_md.conversion.gromacs_files import GromacsFiles
                    gro = tmp / "p.gro"
                    top = tmp / "p.top"
                    itp = tmp / "p.itp"
                    for f in [gro, top, itp]:
                        f.write_text("")
                    mock_save.return_value = GromacsFiles(itp=itp, gro=gro, top=top)

                    pipeline = PolymerParameterisationPipeline(
                        library=_empty_library(),
                        specs=_specs(),
                        n=5,
                        output_dir=tmp,
                    )
                    result = pipeline.run()

                assert isinstance(result, ParameterisedMolecule)
                assert mock_tiler_instance.tile.called

    def test_build_polymer_returns_polymer_instance(self):
        from polymer_md.core.polymer import Polymer

        pipeline = PolymerParameterisationPipeline(
            library=_empty_library(),
            specs=_specs(),
            n=3,
            output_dir=Path("/tmp/test_build_polymer_direct"),
        )
        polymer = pipeline._build_polymer()
        assert isinstance(polymer, Polymer)

    def test_save_gromacs_creates_output_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            structure = pmd.Structure()
            atom = pmd.Atom()
            atom.xx = 0.0
            atom.xy = 0.0
            atom.xz = 0.0
            structure.add_atom(atom, "MOL", 1)

            pipeline = PolymerParameterisationPipeline(
                library=_empty_library(),
                specs=_specs(),
                n=3,
                output_dir=tmp,
            )
            gromacs_files = pipeline._save_gromacs(structure)

            assert gromacs_files.gro.exists()
            assert gromacs_files.top.exists()
            assert gromacs_files.itp.exists()

    def test_adjust_charge_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            mock_polymer = MagicMock()
            mock_polymer.mol = Chem.MolFromSmiles("CCC")
            mock_polymer.residue_instances = []

            with (
                patch.object(PolymerParameterisationPipeline, "_build_polymer", return_value=mock_polymer),
                patch("polymer_md.parameterisation.polymer_pipeline.ETKDGConformerGenerator"),
                patch("polymer_md.parameterisation.polymer_pipeline.TopologyBuilder") as MockTB,
                patch("polymer_md.parameterisation.polymer_pipeline.StructureMolDeriver") as MockDeriv,
                patch("polymer_md.parameterisation.polymer_pipeline.PolymerParameterisationTiler"),
                patch("polymer_md.parameterisation.polymer_pipeline.adjust_charge_neutrality") as mock_adj,
                patch.object(PolymerParameterisationPipeline, "_save_gromacs") as mock_save,
            ):
                mock_structure = MagicMock(spec=pmd.Structure)
                mock_structure.atoms = []
                MockTB.build.return_value = mock_structure
                MockDeriv.derive.return_value = mol

                from polymer_md.conversion.gromacs_files import GromacsFiles
                gro = tmp / "p.gro"; top = tmp / "p.top"; itp = tmp / "p.itp"
                for f in [gro, top, itp]:
                    f.write_text("")
                mock_save.return_value = GromacsFiles(itp=itp, gro=gro, top=top)

                pipeline = PolymerParameterisationPipeline(
                    library=_empty_library(),
                    specs=_specs(),
                    n=3,
                    output_dir=tmp,
                    adjust_charge=False,
                )
                pipeline.run()

            mock_adj.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: ParameterisedMolecule.from_gromacs_files
# ---------------------------------------------------------------------------

class TestParameterisedMoleculeFromGromacsFiles:
    def test_loads_structure_from_gromacs_files(self):
        from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
        from polymer_md.conversion.gromacs_files import GromacsFiles

        mock_structure = MagicMock(spec=pmd.Structure)
        mock_gromacs = MagicMock(spec=GromacsFiles)
        mock_gromacs.top = MagicMock()
        mock_gromacs.gro = MagicMock()
        mock_mol = Chem.MolFromSmiles("CCC")

        with patch("polymer_md.parameterisation.data_models.parameterised_mol.pmd.load_file", return_value=mock_structure) as mock_load:
            result = ParameterisedMolecule.from_gromacs_files(mock_gromacs, mock_mol)

        mock_load.assert_called_once()
        assert result.structure is mock_structure
        assert result.mol is mock_mol
        assert result.source is mock_gromacs

    def test_load_file_called_with_top_and_gro_paths(self):
        from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
        from polymer_md.conversion.gromacs_files import GromacsFiles

        mock_gromacs = MagicMock(spec=GromacsFiles)
        mock_gromacs.top = "/fake/top.top"
        mock_gromacs.gro = "/fake/conf.gro"
        mock_mol = Chem.MolFromSmiles("CCC")

        with patch("polymer_md.parameterisation.data_models.parameterised_mol.pmd.load_file", return_value=MagicMock()) as mock_load:
            ParameterisedMolecule.from_gromacs_files(mock_gromacs, mock_mol)

        call_args = mock_load.call_args
        assert str(mock_gromacs.top) in call_args.args or str(mock_gromacs.top) in str(call_args)
        assert "xyz" in call_args.kwargs or str(mock_gromacs.gro) in str(call_args)
