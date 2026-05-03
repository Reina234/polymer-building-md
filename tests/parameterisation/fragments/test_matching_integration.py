"""
Integration tests for the fragment matching pipeline.

These tests exercise real `StructureMolDeriver.derive()`, `SmartsBuilder`,
`RegionFragmentExtractor`, and `PolymerParameterisationTiler._collect_assignments()`
on small concrete molecules (no acpype required), verifying that:

1. `derive()` preserves the exact bond adjacency of the input mol_3d.
2. Interior fragment SMARTS from a trimer matches at interior positions of a polymer.
3. Terminal fragment SMARTS (with extended context) matches at chain-end positions.
4. No proper dihedral assignments are missing when a real library covers the polymer.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import parmed as pmd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.building.data_models.trimer import BondType as TrimerBondType, TrimerResult
from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.parameters import DihedralParameter
from polymer_md.parameterisation.fragments.extraction.builder import FragmentLibraryBuilder
from polymer_md.parameterisation.fragments.extraction.region_extractor import RegionFragmentExtractor
from polymer_md.parameterisation.tiler import PolymerParameterisationTiler
from polymer_md.utils.parmed_helper import StructureMolDeriver
from polymer_md.utils.topology_builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _embed(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


def _bond_set(mol: Chem.Mol) -> frozenset:
    return frozenset(
        frozenset({b.GetBeginAtomIdx(), b.GetEndAtomIdx()})
        for b in mol.GetBonds()
    )


def _build_typed_structure(mol: Chem.Mol) -> pmd.Structure:
    """Build a parmed structure with dummy parameter types set on all terms."""
    structure = TopologyBuilder.build(mol)
    for atom in structure.atoms:
        z = mol.GetAtomWithIdx(atom.idx).GetAtomicNum()
        atom.charge = -0.05 if z == 6 else 0.05
        atom.epsilon = 0.1
        atom.sigma = 0.35
        atom.mass = 12.0 if z == 6 else 1.008
    for bond in structure.bonds:
        bond.type = pmd.BondType(k=300.0, req=1.52)
    for angle in structure.angles:
        angle.type = pmd.AngleType(k=50.0, theteq=109.5)
    for dihedral in structure.dihedrals:
        dihedral.type = pmd.DihedralType(phi_k=0.5, phase=0.0, per=3)
    return structure


def _identity_mapping(mol: Chem.Mol) -> dict[int, int]:
    """mol3d_to_parmed when TopologyBuilder preserves mol ordering (identity)."""
    return {i: i for i in range(mol.GetNumAtoms())}


def _heavy_indices(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]


def _make_octane_parameterised_trimer() -> ParameterisedTrimer:
    """
    Octane (CH3-CH2-CH2-CH2-CH2-CH2-CH2-CH3) as a trimer:
      cap_left = C0, left = C1+C2, central = C3+C4, right = C5+C6, cap_right = C7
    All bond/angle/dihedral types are set to dummy values so FragmentMatcher
    can extract parameter records without acpype output.
    """
    mol = _embed("CCCCCCCC")
    structure = _build_typed_structure(mol)
    heavy = _heavy_indices(mol)
    trimer_result = TrimerResult(
        left_id="M",
        central_id="M",
        right_id="M",
        left_bond=TrimerBondType(from_site=0, to_site=0),
        right_bond=TrimerBondType(from_site=0, to_site=0),
        mol=mol,
        probability=1.0,
        left_atom_indices=frozenset({heavy[1], heavy[2]}),
        central_atom_indices=frozenset({heavy[3], heavy[4]}),
        right_atom_indices=frozenset({heavy[5], heavy[6]}),
        cap_atom_indices=frozenset({heavy[0], heavy[7]}),
    )
    return ParameterisedTrimer(
        trimer_result=trimer_result,
        gromacs_files=MagicMock(),
        structure=structure,
        mol_3d=mol,
    )


# ---------------------------------------------------------------------------
# 1. derive() topology preservation
# ---------------------------------------------------------------------------

class TestDeriveTopologyPreservation:
    def test_preserves_alkane_bond_adjacency(self):
        mol = _embed("CCC")
        structure = TopologyBuilder.build(mol)
        derived = StructureMolDeriver.derive(structure, mol)
        assert _bond_set(derived) == _bond_set(mol)

    def test_preserves_ester_bond_adjacency(self):
        """Ester group has C=O and C-O-C; coordinate-based methods can mis-assign these."""
        mol = _embed("CC(=O)OCC")
        structure = TopologyBuilder.build(mol)
        derived = StructureMolDeriver.derive(structure, mol)
        assert _bond_set(derived) == _bond_set(mol)

    def test_preserves_aromatic_ring_bond_adjacency(self):
        mol = _embed("c1ccccc1")
        structure = TopologyBuilder.build(mol)
        derived = StructureMolDeriver.derive(structure, mol)
        assert _bond_set(derived) == _bond_set(mol)

    def test_derived_atom_count_equals_original(self):
        mol = _embed("CCCC")
        structure = TopologyBuilder.build(mol)
        derived = StructureMolDeriver.derive(structure, mol)
        assert derived.GetNumAtoms() == mol.GetNumAtoms()

    def test_derived_mol_smarts_matches_itself(self):
        """SMARTS built from derived mol must match the derived mol (sanity check)."""
        from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder

        mol = _embed("CCCC")
        structure = TopologyBuilder.build(mol)
        derived = StructureMolDeriver.derive(structure, mol)
        heavy = _heavy_indices(mol)
        smarts, _ = SmartsBuilder.subgraph(derived, tuple(heavy))
        query = Chem.MolFromSmarts(smarts)
        assert query is not None
        matches = derived.GetSubstructMatches(query)
        assert len(matches) >= 1


# ---------------------------------------------------------------------------
# 2. Interior fragment SMARTS matching
# ---------------------------------------------------------------------------

class TestInteriorFragmentSmartsMatchesPolymer:
    def _interior_fragment(self, trimer_mol: Chem.Mol, trimer_structure: pmd.Structure):
        trimer_derived = StructureMolDeriver.derive(trimer_structure, trimer_mol)
        id_map = _identity_mapping(trimer_mol)
        extractor = RegionFragmentExtractor()
        heavy = _heavy_indices(trimer_mol)
        left = extractor.resolve_parmed_indices(frozenset({heavy[1], heavy[2]}), trimer_mol, id_map)
        central = extractor.resolve_parmed_indices(frozenset({heavy[3], heavy[4]}), trimer_mol, id_map)
        right = extractor.resolve_parmed_indices(frozenset({heavy[5], heavy[6]}), trimer_mol, id_map)
        return extractor.extract(
            derived_mol=trimer_derived,
            structure=trimer_structure,
            context_parmed_indices=left | central | right,
            region_parmed_indices=central,
        )

    def test_interior_smarts_matches_in_longer_chain(self):
        trimer_mol = _embed("CCCCCCCC")
        trimer_structure = TopologyBuilder.build(trimer_mol)
        fragment, _ = self._interior_fragment(trimer_mol, trimer_structure)

        polymer_mol = _embed("CCCCCCCCCC")
        polymer_structure = TopologyBuilder.build(polymer_mol)
        polymer_derived = StructureMolDeriver.derive(polymer_structure, polymer_mol)

        query = Chem.MolFromSmarts(fragment.pattern)
        assert query is not None, "Interior fragment produced invalid SMARTS"
        matches = polymer_derived.GetSubstructMatches(query)
        assert len(matches) > 0, (
            f"Interior fragment SMARTS did not match in polymer; pattern={fragment.pattern!r}"
        )

    def test_interior_fragment_has_dihedral_members(self):
        trimer_mol = _embed("CCCCCCCC")
        trimer_structure = TopologyBuilder.build(trimer_mol)
        fragment, _ = self._interior_fragment(trimer_mol, trimer_structure)
        assert len(fragment.annotated_dihedrals) > 0, (
            "Interior fragment must contain dihedral annotated members"
        )

    def test_interior_fragment_context_excludes_cap_atoms(self):
        """Cap atoms (C0 and C7 of octane) must not appear in interior SMARTS pattern."""
        trimer_mol = _embed("CCCCCCCC")
        trimer_structure = _build_typed_structure(trimer_mol)
        fragment, global_to_local = self._interior_fragment(trimer_mol, trimer_structure)
        heavy = _heavy_indices(trimer_mol)
        cap_parmed_indices = {heavy[0], heavy[7]}
        for cap_idx in cap_parmed_indices:
            assert cap_idx not in global_to_local, (
                f"Cap atom (parmed idx {cap_idx}) must not appear in interior fragment"
            )


# ---------------------------------------------------------------------------
# 3. Terminal fragment covers cap-adjacent dihedrals
# ---------------------------------------------------------------------------

class TestTerminalFragmentCoversCapDihedrals:
    def _left_terminal_fragment(
        self,
        trimer_mol: Chem.Mol,
        trimer_structure: pmd.Structure,
        include_central: bool,
    ):
        """Extract left terminal fragment with or without the central monomer in context."""
        trimer_derived = StructureMolDeriver.derive(trimer_structure, trimer_mol)
        id_map = _identity_mapping(trimer_mol)
        extractor = RegionFragmentExtractor()
        heavy = _heavy_indices(trimer_mol)
        cap_left = extractor.resolve_parmed_indices(frozenset({heavy[0]}), trimer_mol, id_map)
        left = extractor.resolve_parmed_indices(frozenset({heavy[1], heavy[2]}), trimer_mol, id_map)
        central = extractor.resolve_parmed_indices(frozenset({heavy[3], heavy[4]}), trimer_mol, id_map)
        context = cap_left | left | (central if include_central else frozenset())
        return extractor.extract(
            derived_mol=trimer_derived,
            structure=trimer_structure,
            context_parmed_indices=context,
            region_parmed_indices=cap_left | left,
        )

    def test_left_terminal_with_central_has_more_dihedrals_than_without(self):
        """Including the central monomer in context must capture spanning dihedrals."""
        trimer_mol = _embed("CCCCCCCC")
        trimer_structure = _build_typed_structure(trimer_mol)
        frag_with, _ = self._left_terminal_fragment(trimer_mol, trimer_structure, include_central=True)
        frag_without, _ = self._left_terminal_fragment(trimer_mol, trimer_structure, include_central=False)
        assert len(frag_with.annotated_dihedrals) > len(frag_without.annotated_dihedrals), (
            "Terminal fragment with central in context must capture more dihedrals "
            "than one without central"
        )

    def test_left_terminal_smarts_matches_at_chain_end_of_longer_molecule(self):
        trimer_mol = _embed("CCCCCCCC")
        trimer_structure = TopologyBuilder.build(trimer_mol)
        fragment, _ = self._left_terminal_fragment(trimer_mol, trimer_structure, include_central=True)

        polymer_mol = _embed("CCCCCCCCCC")
        polymer_structure = TopologyBuilder.build(polymer_mol)
        polymer_derived = StructureMolDeriver.derive(polymer_structure, polymer_mol)

        query = Chem.MolFromSmarts(fragment.pattern)
        assert query is not None, "Left terminal fragment produced invalid SMARTS"
        matches = polymer_derived.GetSubstructMatches(query)
        assert len(matches) > 0, (
            "Left terminal fragment SMARTS did not match at chain end of polymer"
        )


# ---------------------------------------------------------------------------
# 4. Full dihedral coverage — no missing assignments from a real library
# ---------------------------------------------------------------------------

class TestDihedralCoverageFromRealLibrary:
    def test_no_missing_proper_dihedrals_in_polymer(self):
        """
        Build a fragment library from an octane trimer (TopologyBuilder, no acpype).
        Verify that every proper dihedral in a decane polymer structure has an
        assignment entry produced by _collect_assignments.

        This is the canonical regression test for the DetermineBonds topology-mismatch
        bug and the terminal-fragment missing-central-context bug.
        """
        parameterised_trimer = _make_octane_parameterised_trimer()
        library = FragmentLibraryBuilder().build([parameterised_trimer])
        assert len(library.records) > 0, "Library must have at least one record"

        polymer_mol = _embed("CCCCCCCCCC")
        polymer_structure = TopologyBuilder.build(polymer_mol)
        polymer_derived = StructureMolDeriver.derive(polymer_structure, polymer_mol)

        tiler = PolymerParameterisationTiler(library=library, missing_strategies={})
        assignments = tiler._collect_assignments(polymer_derived)

        missing = []
        for dihedral in polymer_structure.dihedrals:
            if dihedral.improper:
                continue
            i = dihedral.atom1.idx
            j = dihedral.atom2.idx
            k = dihedral.atom3.idx
            l = dihedral.atom4.idx
            canonical = min((i, j, k, l), (l, k, j, i))
            if (canonical, DihedralParameter.FORCE_CONSTANT) not in assignments:
                missing.append(canonical)

        assert missing == [], (
            f"{len(missing)} proper dihedral(s) have no library assignment: "
            f"{missing[:5]!r}"
        )

    def test_library_has_both_interior_and_terminal_records(self):
        """Library must produce records from both interior and terminal fragments."""
        parameterised_trimer = _make_octane_parameterised_trimer()
        library = FragmentLibraryBuilder().build([parameterised_trimer])
        assert len(library.records) > 0

        patterns = {hit.fragment.pattern for r in library.records for hit in r.hits}
        assert len(patterns) >= 2, (
            "Library should have at least 2 distinct fragment patterns "
            "(one interior, at least one terminal)"
        )
