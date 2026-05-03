from __future__ import annotations

import json

import pytest

from polymer_md.analysis.results import ComparisonResult, Summary
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ImproperParameter,
)

_FRAG = Fragment(pattern="[#6;A]-[#6;A]")


class TestSummary:
    def test_fields_accessible(self):
        s = Summary(mean=1.0, std=0.5, count=3, min=0.5, max=1.5)
        assert s.mean == pytest.approx(1.0)
        assert s.std == pytest.approx(0.5)
        assert s.count == 3
        assert s.min == pytest.approx(0.5)
        assert s.max == pytest.approx(1.5)


class TestComparisonResultSummarise:
    def test_summarise_correct_mean(self):
        results = {
            "mol_a": {_FRAG: {AtomParameter.CHARGE: [0.1, 0.3]}}
        }
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        assert pytest.approx(summary["mol_a"][_FRAG][AtomParameter.CHARGE].mean, abs=1e-6) == 0.2

    def test_summarise_correct_count(self):
        results = {"mol_a": {_FRAG: {BondParameter.FORCE_CONSTANT: [300.0, 310.0, 290.0]}}}
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        assert summary["mol_a"][_FRAG][BondParameter.FORCE_CONSTANT].count == 3

    def test_summarise_single_value_std_zero(self):
        results = {"mol_a": {_FRAG: {AtomParameter.CHARGE: [0.5]}}}
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        assert summary["mol_a"][_FRAG][AtomParameter.CHARGE].std == pytest.approx(0.0)

    def test_summarise_two_values_std(self):
        import statistics
        values = [1.0, 3.0]
        results = {"mol_a": {_FRAG: {AtomParameter.CHARGE: values}}}
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        assert summary["mol_a"][_FRAG][AtomParameter.CHARGE].std == pytest.approx(statistics.stdev(values))

    def test_summarise_correct_min_max(self):
        results = {"mol_a": {_FRAG: {AtomParameter.CHARGE: [-0.5, 0.0, 0.5]}}}
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        s = summary["mol_a"][_FRAG][AtomParameter.CHARGE]
        assert s.min == pytest.approx(-0.5)
        assert s.max == pytest.approx(0.5)

    def test_empty_values_list_excluded_from_summary(self):
        results = {"mol_a": {_FRAG: {AtomParameter.CHARGE: []}}}
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        assert AtomParameter.CHARGE not in summary["mol_a"].get(_FRAG, {})

    def test_multiple_labels(self):
        results = {
            "a": {_FRAG: {AtomParameter.CHARGE: [0.1]}},
            "b": {_FRAG: {AtomParameter.CHARGE: [0.2]}},
        }
        comparison = ComparisonResult(results=results)
        summary = comparison.summarise()
        assert "a" in summary
        assert "b" in summary

    def test_empty_results(self):
        comparison = ComparisonResult(results={})
        assert comparison.summarise() == {}


class TestComparisonResultSaveLoad:
    def test_round_trip_preserves_labels(self, tmp_path):
        frag = Fragment(pattern="[#6;A]-[#6;A]")
        results = {"mol_a": {frag: {BondParameter.FORCE_CONSTANT: [230.1, 231.5]}}}
        comparison = ComparisonResult(results=results)
        path = tmp_path / "results.json"
        comparison.save(path)
        loaded = ComparisonResult.load(path)
        assert set(loaded.results.keys()) == {"mol_a"}

    def test_round_trip_preserves_values(self, tmp_path):
        frag = Fragment(pattern="[#6;A]-[#6;A]")
        values = [230.1, 231.5, 229.8]
        results = {"mol_a": {frag: {BondParameter.FORCE_CONSTANT: values}}}
        comparison = ComparisonResult(results=results)
        comparison.save(tmp_path / "r.json")
        loaded = ComparisonResult.load(tmp_path / "r.json")
        loaded_frag = Fragment(pattern="[#6;A]-[#6;A]")
        assert loaded.results["mol_a"][loaded_frag][BondParameter.FORCE_CONSTANT] == pytest.approx(values)

    def test_round_trip_multiple_parameters(self, tmp_path):
        frag = Fragment(pattern="[#6;A]-[#6;A]")
        results = {
            "mol_a": {
                frag: {
                    BondParameter.FORCE_CONSTANT: [230.0],
                    BondParameter.EQUILIBRIUM_LENGTH: [1.534],
                }
            }
        }
        comparison = ComparisonResult(results=results)
        comparison.save(tmp_path / "r.json")
        loaded = ComparisonResult.load(tmp_path / "r.json")
        loaded_frag = Fragment(pattern="[#6;A]-[#6;A]")
        assert BondParameter.FORCE_CONSTANT in loaded.results["mol_a"][loaded_frag]
        assert BondParameter.EQUILIBRIUM_LENGTH in loaded.results["mol_a"][loaded_frag]

    def test_round_trip_all_parameter_classes(self, tmp_path):
        frag = Fragment(pattern="[#6;A]")
        results = {
            "mol": {
                frag: {
                    AtomParameter.CHARGE: [0.1],
                    AngleParameter.FORCE_CONSTANT: [50.0],
                    DihedralParameter.FORCE_CONSTANT: [1.5],
                    ImproperParameter.FORCE_CONSTANT: [1.1],
                }
            }
        }
        comparison = ComparisonResult(results=results)
        comparison.save(tmp_path / "r.json")
        loaded = ComparisonResult.load(tmp_path / "r.json")
        loaded_frag = Fragment(pattern="[#6;A]")
        param_map = loaded.results["mol"][loaded_frag]
        assert AtomParameter.CHARGE in param_map
        assert AngleParameter.FORCE_CONSTANT in param_map
        assert DihedralParameter.FORCE_CONSTANT in param_map
        assert ImproperParameter.FORCE_CONSTANT in param_map

    def test_round_trip_multiple_molecules(self, tmp_path):
        frag = Fragment(pattern="[#6;A]")
        results = {
            "seed_1": {frag: {AtomParameter.CHARGE: [0.1, 0.2]}},
            "seed_2": {frag: {AtomParameter.CHARGE: [0.15, 0.25]}},
        }
        comparison = ComparisonResult(results=results)
        comparison.save(tmp_path / "r.json")
        loaded = ComparisonResult.load(tmp_path / "r.json")
        assert set(loaded.results.keys()) == {"seed_1", "seed_2"}

    def test_save_creates_parent_directories(self, tmp_path):
        frag = Fragment(pattern="[#6;A]")
        results = {"mol": {frag: {AtomParameter.CHARGE: [0.1]}}}
        comparison = ComparisonResult(results=results)
        nested_path = tmp_path / "deep" / "nested" / "results.json"
        comparison.save(nested_path)
        assert nested_path.exists()

    def test_load_rejects_unknown_schema_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": "99", "results": {}}))
        with pytest.raises(ValueError, match="schema version"):
            ComparisonResult.load(path)

    def test_load_unknown_parameter_key_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "schema_version": "1",
            "results": {"mol": {"[#6]": {"UnknownClass.FORCE_CONSTANT": [1.0]}}},
        }))
        with pytest.raises(ValueError, match="Unknown parameter key"):
            ComparisonResult.load(path)

    def test_saved_json_is_human_readable(self, tmp_path):
        frag = Fragment(pattern="[#6;A]-[#6;A]")
        results = {"mol_a": {frag: {BondParameter.FORCE_CONSTANT: [230.0]}}}
        comparison = ComparisonResult(results=results)
        path = tmp_path / "r.json"
        comparison.save(path)
        raw = json.loads(path.read_text())
        assert "schema_version" in raw
        assert "results" in raw
        assert "mol_a" in raw["results"]
        assert "[#6;A]-[#6;A]" in raw["results"]["mol_a"]
        assert "BondParameter.FORCE_CONSTANT" in raw["results"]["mol_a"]["[#6;A]-[#6;A]"]
