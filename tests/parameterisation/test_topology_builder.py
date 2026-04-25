from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.utils.topology_builder import TopologyBuilder


@pytest.fixture
def ethane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def propane_mol_3d() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def methane_no_conformer() -> Chem.Mol:
    return Chem.AddHs(Chem.MolFromSmiles("C"))


class TestTopologyBuilderAtoms:
    def test_atom_count_matches_mol(self, ethane_mol):
        structure = TopologyBuilder.build(ethane_mol)
        assert len(structure.atoms) == ethane_mol.GetNumAtoms()

    def test_atom_names_use_symbol_and_index(self, ethane_mol):
        structure = TopologyBuilder.build(ethane_mol)
        for i, atom in enumerate(structure.atoms):
            rdkit_atom = ethane_mol.GetAtomWithIdx(i)
            assert atom.name == f"{rdkit_atom.GetSymbol()}{i}"

    def test_atom_mass_is_positive(self, ethane_mol):
        structure = TopologyBuilder.build(ethane_mol)
        for atom in structure.atoms:
            assert atom.mass > 0

    def test_coordinates_set_from_conformer(self, ethane_mol):
        structure = TopologyBuilder.build(ethane_mol)
        conformer = ethane_mol.GetConformer()
        for i, atom in enumerate(structure.atoms):
            pos = conformer.GetAtomPosition(i)
            assert abs(atom.xx - pos.x) < 1e-6
            assert abs(atom.xy - pos.y) < 1e-6
            assert abs(atom.xz - pos.z) < 1e-6

    def test_no_conformer_does_not_raise(self, methane_no_conformer):
        structure = TopologyBuilder.build(methane_no_conformer)
        assert len(structure.atoms) == methane_no_conformer.GetNumAtoms()

    def test_atoms_indexed_sequentially(self, propane_mol_3d):
        structure = TopologyBuilder.build(propane_mol_3d)
        for i, atom in enumerate(structure.atoms):
            assert atom.idx == i


class TestTopologyBuilderBonds:
    def test_bond_count_matches_mol(self, ethane_mol):
        structure = TopologyBuilder.build(ethane_mol)
        assert len(structure.bonds) == ethane_mol.GetNumBonds()

    def test_bond_atoms_match_mol_bonds(self, ethane_mol):
        structure = TopologyBuilder.build(ethane_mol)
        mol_bonds = {
            frozenset({b.GetBeginAtomIdx(), b.GetEndAtomIdx()})
            for b in ethane_mol.GetBonds()
        }
        pmd_bonds = {
            frozenset({b.atom1.idx, b.atom2.idx})
            for b in structure.bonds
        }
        assert mol_bonds == pmd_bonds


class TestTopologyBuilderAngles:
    def test_angles_present_for_propane(self, propane_mol_3d):
        structure = TopologyBuilder.build(propane_mol_3d)
        assert len(structure.angles) > 0

    def test_angle_atoms_all_come_from_mol(self, propane_mol_3d):
        structure = TopologyBuilder.build(propane_mol_3d)
        n_atoms = propane_mol_3d.GetNumAtoms()
        for angle in structure.angles:
            assert angle.atom1.idx < n_atoms
            assert angle.atom2.idx < n_atoms
            assert angle.atom3.idx < n_atoms

    def test_no_duplicate_angles(self, propane_mol_3d):
        structure = TopologyBuilder.build(propane_mol_3d)
        seen = set()
        for angle in structure.angles:
            i, j, k = angle.atom1.idx, angle.atom2.idx, angle.atom3.idx
            key = (min(i, k), j, max(i, k))
            assert key not in seen, f"Duplicate angle: {key}"
            seen.add(key)

    def test_ethane_has_no_angles(self, ethane_mol):
        mol = Chem.MolFromSmiles("CC")
        structure = TopologyBuilder.build(mol)
        assert len(structure.angles) == 0


class TestTopologyBuilderDihedrals:
    def test_dihedrals_present_for_butane(self):
        mol = Chem.MolFromSmiles("CCCC")
        structure = TopologyBuilder.build(mol)
        assert len(structure.dihedrals) > 0

    def test_no_duplicate_dihedrals(self, propane_mol_3d):
        structure = TopologyBuilder.build(propane_mol_3d)
        seen = set()
        for dihedral in structure.dihedrals:
            forward = (dihedral.atom1.idx, dihedral.atom2.idx, dihedral.atom3.idx, dihedral.atom4.idx)
            reverse = forward[::-1]
            key = min(forward, reverse)
            assert key not in seen, f"Duplicate dihedral: {key}"
            seen.add(key)

    def test_propane_has_no_dihedrals(self):
        mol = Chem.MolFromSmiles("CCC")
        structure = TopologyBuilder.build(mol)
        assert len(structure.dihedrals) == 0

    def test_dihedral_atoms_form_connected_chain(self):
        mol = Chem.MolFromSmiles("CCCC")
        structure = TopologyBuilder.build(mol)
        bond_set = {
            frozenset({b.GetBeginAtomIdx(), b.GetEndAtomIdx()})
            for b in mol.GetBonds()
        }
        for dihedral in structure.dihedrals:
            i = dihedral.atom1.idx
            j = dihedral.atom2.idx
            k = dihedral.atom3.idx
            l = dihedral.atom4.idx
            assert frozenset({i, j}) in bond_set
            assert frozenset({j, k}) in bond_set
            assert frozenset({k, l}) in bond_set

    def test_cyclopropane_has_no_dihedrals_due_to_shared_neighbors(self):
        mol = Chem.MolFromSmiles("C1CC1")
        structure = TopologyBuilder.build(mol)
        assert len(structure.dihedrals) == 0
