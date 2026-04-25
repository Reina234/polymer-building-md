from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.parameterisation.fragment_library_pipeline import FragmentLibraryPipeline
from polymer_md.parameterisation.fragments.library import FragmentLibrary


def _make_mock_library() -> FragmentLibrary:
    return FragmentLibrary(records=(), atom_metadata={})


def _make_specs() -> list[MonomerSpec]:
    from polymer_md.building.monomer_converter import MonomerToResidueConverter
    from polymer_md.core.monomer import Monomer
    residue = MonomerToResidueConverter.convert(Monomer(smiles="C=Cc1ccccc1", label="S"))
    return [MonomerSpec(residue=residue, weight=1.0)]


class TestFragmentLibraryPipeline:
    def test_run_calls_trimer_pipeline_and_builder(self):
        specs = _make_specs()
        mock_library = _make_mock_library()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            library_path = tmp / "library.json"

            with (
                patch(
                    "polymer_md.parameterisation.fragment_library_pipeline.TrimerParameterisationPipeline"
                ) as MockTrimerPipeline,
                patch(
                    "polymer_md.parameterisation.fragment_library_pipeline.FragmentLibraryBuilder"
                ) as MockBuilder,
            ):
                mock_trimer_instance = MagicMock()
                mock_trimer_instance.run.return_value = []
                MockTrimerPipeline.return_value = mock_trimer_instance

                mock_builder_instance = MagicMock()
                mock_builder_instance.build.return_value = mock_library
                MockBuilder.return_value = mock_builder_instance

                pipeline = FragmentLibraryPipeline(
                    specs=specs,
                    output_dir=tmp / "trimers",
                    library_path=library_path,
                )
                result = pipeline.run()

            assert mock_trimer_instance.run.called
            assert mock_builder_instance.build.called
            assert library_path.exists()
            assert result is mock_library

    def test_run_saves_library_to_disk(self):
        specs = _make_specs()
        mock_library = FragmentLibrary(records=(), atom_metadata={})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            library_path = tmp / "library.json"

            with (
                patch("polymer_md.parameterisation.fragment_library_pipeline.TrimerParameterisationPipeline") as MockTP,
                patch("polymer_md.parameterisation.fragment_library_pipeline.FragmentLibraryBuilder") as MockB,
            ):
                MockTP.return_value.run.return_value = []
                MockB.return_value.build.return_value = mock_library

                pipeline = FragmentLibraryPipeline(
                    specs=specs,
                    output_dir=tmp / "trimers",
                    library_path=library_path,
                )
                pipeline.run()

            assert library_path.exists()
            data = json.loads(library_path.read_text())
            assert "records" in data

    def test_run_returns_library_from_builder(self):
        specs = _make_specs()
        mock_library = FragmentLibrary(records=(), atom_metadata={"pat": {}})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            with (
                patch("polymer_md.parameterisation.fragment_library_pipeline.TrimerParameterisationPipeline") as MockTP,
                patch("polymer_md.parameterisation.fragment_library_pipeline.FragmentLibraryBuilder") as MockB,
            ):
                MockTP.return_value.run.return_value = []
                MockB.return_value.build.return_value = mock_library

                pipeline = FragmentLibraryPipeline(
                    specs=specs,
                    output_dir=tmp / "trimers",
                    library_path=tmp / "lib.json",
                )
                result = pipeline.run()

            assert result is mock_library

    def test_pipeline_passes_probability_threshold_to_trimer_pipeline(self):
        specs = _make_specs()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            with (
                patch("polymer_md.parameterisation.fragment_library_pipeline.TrimerParameterisationPipeline") as MockTP,
                patch("polymer_md.parameterisation.fragment_library_pipeline.FragmentLibraryBuilder") as MockB,
            ):
                MockTP.return_value.run.return_value = []
                MockB.return_value.build.return_value = _make_mock_library()

                pipeline = FragmentLibraryPipeline(
                    specs=specs,
                    output_dir=tmp / "trimers",
                    library_path=tmp / "lib.json",
                    probability_threshold=0.05,
                )
                pipeline.run()

            call_kwargs = MockTP.call_args.kwargs
            assert call_kwargs["probability_threshold"] == 0.05
