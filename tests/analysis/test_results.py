from __future__ import annotations

import pytest

from polymer_md.analysis.results import ComparisonResult, Summary
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter

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
