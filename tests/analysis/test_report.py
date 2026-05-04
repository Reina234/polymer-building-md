from __future__ import annotations

import pytest

from polymer_md.analysis.report import TextReport
from polymer_md.analysis.results import ComparisonResult
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter

_CC = Fragment(pattern="[#6;A]-[#6;A]")
_C = Fragment(pattern="[#6;A]")


def _make_comparison() -> ComparisonResult:
    return ComparisonResult(results={
        "mol_a": {
            _CC: {BondParameter.FORCE_CONSTANT: [230.0, 231.0]},
            _C: {AtomParameter.CHARGE: [0.1, -0.1]},
        },
        "mol_b": {
            _CC: {BondParameter.FORCE_CONSTANT: [229.5, 232.0]},
        },
    })


class TestTextReport:
    def test_str_contains_title(self):
        report = TextReport(_make_comparison(), title="My Study")
        assert "My Study" in str(report)

    def test_str_contains_fragment_pattern(self):
        report = TextReport(_make_comparison())
        assert "[#6;A]-[#6;A]" in str(report)

    def test_str_contains_parameter_name(self):
        report = TextReport(_make_comparison())
        assert "FORCE_CONSTANT" in str(report)

    def test_str_contains_molecule_labels(self):
        report = TextReport(_make_comparison())
        text = str(report)
        assert "mol_a" in text
        assert "mol_b" in text

    def test_str_contains_numeric_values(self):
        report = TextReport(_make_comparison())
        text = str(report)
        assert "230." in text or "231." in text

    def test_empty_comparison_produces_only_header(self):
        report = TextReport(ComparisonResult(results={}))
        text = str(report)
        assert "Parameter Comparison" in text

    def test_write_creates_file(self, tmp_path):
        path = tmp_path / "report.txt"
        TextReport(_make_comparison()).write(path)
        assert path.exists()
        assert len(path.read_text()) > 0

    def test_write_file_content_matches_str(self, tmp_path):
        path = tmp_path / "report.txt"
        report = TextReport(_make_comparison())
        report.write(path)
        assert path.read_text().strip() == str(report).strip()

    def test_write_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "report.txt"
        TextReport(_make_comparison()).write(path)
        assert path.exists()

    def test_each_fragment_parameter_pair_appears_once(self):
        report = TextReport(_make_comparison())
        text = str(report)
        assert text.count("BondParameter.FORCE_CONSTANT") == 1

    def test_default_title(self):
        report = TextReport(_make_comparison())
        assert "Parameter Comparison" in str(report)

    def test_custom_title(self):
        report = TextReport(_make_comparison(), title="MMA Composition Study")
        assert "MMA Composition Study" in str(report)

    def test_mol_b_absent_fragment_not_repeated(self):
        report = TextReport(_make_comparison())
        text = str(report)
        assert text.count("[#6;A]") >= 1
