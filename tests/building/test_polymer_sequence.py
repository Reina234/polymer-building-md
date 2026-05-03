"""Regression tests for physical bond semantics in RandomPolymerBuilder and oriented-triplet extraction.

The key invariant: connections[i] = (from_site_index, to_site_index) is the physical bond
between monomer i and monomer i+1. For an HT polymer:
  - connections[0] is a TT bond (TAIL of first monomer → TAIL of second)
  - connections[1:] are HT bonds (HEAD of current growing end → TAIL of incoming)

Before the fix, connections[0][0] was set to active_site.site_index (=HEAD=0) instead of
initial_site.site_index (=TAIL=1), producing the wrong trimer orientation for the first
inter-monomer context.
"""
from __future__ import annotations

import numpy as np
import pytest

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.data_models.trimer import BondType
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.random_polymer import RandomPolymerBuilder
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.building.trimer_builder import TrimerBuilder
from polymer_md.core.caps import BuiltinCap
from polymer_md.core.monomer import Monomer
from polymer_md.core.residue_instance import ResidueType
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline


_MONOMER_SMILES = "C=CC"
_MONOMER_LABEL = "P"
_HEAD = 0
_TAIL = 1


def _make_residue():
    return MonomerToResidueConverter.convert(Monomer(smiles=_MONOMER_SMILES, label=_MONOMER_LABEL))


def _make_spec(weight: float = 1.0) -> MonomerSpec:
    return MonomerSpec(residue=_make_residue(), weight=weight)


def _make_builder(ht_fraction: float) -> tuple[RandomPolymerBuilder, dict]:
    residue = _make_residue()
    spec = MonomerSpec(residue=residue, weight=1.0)
    matrix = ProportionalSolver(ht_fraction=ht_fraction).solve([spec])
    residues = {residue.id: residue}
    builder = RandomPolymerBuilder(residues=residues, matrix=matrix, cap=BuiltinCap.METHYL)
    return builder, residues


def _build(n: int, ht_fraction: float) -> tuple:
    builder, _ = _make_builder(ht_fraction)
    rng = np.random.default_rng(42)
    return builder.build_with_connections(n, rng)


# ---------------------------------------------------------------------------
# Physical bond semantics for connections
# ---------------------------------------------------------------------------


def test_first_connection_is_tt_for_fully_ht_polymer():
    """For ht_fraction=1.0, the chain starts at TAIL, so the first inter-monomer bond
    is TAIL→TAIL (TT), not HEAD→TAIL (HT)."""
    _, connections = _build(n=4, ht_fraction=1.0)
    from_site, to_site = connections[0]
    assert from_site == _TAIL
    assert to_site == _TAIL


def test_subsequent_connections_are_ht_for_fully_ht_polymer():
    """After the first TT bond, every subsequent bond is HT for ht_fraction=1.0."""
    _, connections = _build(n=5, ht_fraction=1.0)
    for from_site, to_site in connections[1:]:
        assert from_site == _HEAD
        assert to_site == _TAIL


def test_connection_count_equals_n_minus_one():
    """There are exactly n-1 connections for an n-mer polymer."""
    n = 6
    _, connections = _build(n=n, ht_fraction=1.0)
    assert len(connections) == n - 1


def test_first_connection_is_hh_for_fully_hh_polymer():
    """For ht_fraction=0.0, the chain starts at HEAD, so the first bond is HEAD→HEAD (HH)."""
    _, connections = _build(n=4, ht_fraction=0.0)
    from_site, to_site = connections[0]
    assert from_site == _HEAD
    assert to_site == _HEAD


# ---------------------------------------------------------------------------
# Oriented-triplet extraction
# ---------------------------------------------------------------------------


def test_extract_oriented_triplets_returns_one_triplet_for_three_mer():
    """A 3-mer has only one interior monomer so exactly one oriented triplet is extracted."""
    polymer, connections = _build(n=3, ht_fraction=1.0)
    triplets = PolymerParameterisationPipeline._extract_oriented_triplets(polymer, connections)
    assert len(triplets) == 1


def test_extract_oriented_triplets_returns_two_distinct_types_for_four_mer_ht():
    """For an HT homopolymer of length 4, the first triplet is TT-HT and the second is HT-HT.

    These are chemically distinct environments because the first inter-monomer bond is TT,
    not HT like the rest of the chain.
    """
    polymer, connections = _build(n=4, ht_fraction=1.0)
    triplets = PolymerParameterisationPipeline._extract_oriented_triplets(polymer, connections)

    tt_ht = (_MONOMER_LABEL, _TAIL, _MONOMER_LABEL, _TAIL, _MONOMER_LABEL, _TAIL)
    ht_ht = (_MONOMER_LABEL, _HEAD, _MONOMER_LABEL, _TAIL, _MONOMER_LABEL, _TAIL)
    assert tt_ht in triplets
    assert ht_ht in triplets


def test_extract_oriented_triplets_skips_cap_residues():
    """Cap residues are excluded: a 4-mer polymer with 2 caps has 4 monomers, yielding
    at most 3 interior positions (i=1,2,3), so at most 3 distinct triplets (possibly fewer
    after deduplication).
    """
    polymer, connections = _build(n=4, ht_fraction=1.0)
    non_cap = [r for r in polymer.residue_instances if r.residue_type != ResidueType.CAP]
    triplets = PolymerParameterisationPipeline._extract_oriented_triplets(polymer, connections)
    assert len(non_cap) == 4
    assert len(triplets) <= 3


def test_oriented_triplets_grow_with_longer_polymer():
    """A longer polymer introduces more potentially distinct triplet contexts."""
    _, connections_short = _build(n=4, ht_fraction=1.0)
    polymer_short, _ = _build(n=4, ht_fraction=1.0)
    polymer_long, connections_long = _build(n=10, ht_fraction=1.0)

    triplets_short = PolymerParameterisationPipeline._extract_oriented_triplets(
        polymer_short, connections_short
    )
    triplets_long = PolymerParameterisationPipeline._extract_oriented_triplets(
        polymer_long, connections_long
    )
    assert len(triplets_long) >= len(triplets_short)


# ---------------------------------------------------------------------------
# Trimer building from oriented triplets
# ---------------------------------------------------------------------------


def test_build_for_oriented_triplets_produces_tt_ht_and_ht_ht_trimers():
    """For an HT homopolymer, the two oriented triplets must yield trimers with
    TT-HT and HT-HT bond types respectively — not the same HT-HT trimer twice."""
    polymer, connections = _build(n=4, ht_fraction=1.0)
    triplets = PolymerParameterisationPipeline._extract_oriented_triplets(polymer, connections)

    residue = _make_residue()
    matrix = ProportionalSolver(ht_fraction=1.0).solve([_make_spec()])
    trimers = TrimerBuilder(
        residues={residue.id: residue}, matrix=matrix, cap=BuiltinCap.METHYL
    ).build_for_oriented_triplets(triplets)

    left_bond_labels = {t.left_bond.label for t in trimers}
    assert "TT" in left_bond_labels, "Expected a TT-left-bond trimer for the first inter-monomer bond"
    assert "HT" in left_bond_labels, "Expected an HT-left-bond trimer for interior bonds"


def test_build_for_oriented_triplets_deduplicates_by_canonical_smiles():
    """Two oriented triplets that produce the same molecule are stored only once."""
    polymer, connections = _build(n=10, ht_fraction=1.0)
    triplets = PolymerParameterisationPipeline._extract_oriented_triplets(polymer, connections)

    residue = _make_residue()
    matrix = ProportionalSolver(ht_fraction=1.0).solve([_make_spec()])
    trimers = TrimerBuilder(
        residues={residue.id: residue}, matrix=matrix, cap=BuiltinCap.METHYL
    ).build_for_oriented_triplets(triplets)

    assert len(trimers) <= len(triplets)


def test_build_for_oriented_triplets_all_have_same_monomer_label():
    """All trimers built from a single-monomer polymer reference that monomer in all positions."""
    polymer, connections = _build(n=6, ht_fraction=1.0)
    triplets = PolymerParameterisationPipeline._extract_oriented_triplets(polymer, connections)

    residue = _make_residue()
    matrix = ProportionalSolver(ht_fraction=1.0).solve([_make_spec()])
    trimers = TrimerBuilder(
        residues={residue.id: residue}, matrix=matrix, cap=BuiltinCap.METHYL
    ).build_for_oriented_triplets(triplets)

    for trimer in trimers:
        assert trimer.left_id == _MONOMER_LABEL
        assert trimer.central_id == _MONOMER_LABEL
        assert trimer.right_id == _MONOMER_LABEL
