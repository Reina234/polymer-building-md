from __future__ import annotations

import pytest
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAtom, AnnotatedBond, AnnotatedAngle, AnnotatedDihedral,
)
from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterHit, ParameterRecord
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AtomParameter, BondParameter, AngleParameter, DihedralParameter,
)
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext
from polymer_md.parameterisation.strategies.strict import StrictMissingParameterStrategy
from polymer_md.parameterisation.tiler import PolymerParameterisationTiler, adjust_charge_neutrality
from polymer_md.utils.topology_builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(Chem.AddHs(mol), AllChem.ETKDGv3())
    # Return the mol with H atoms matching topology
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    return mol_h


def _make_library_with_atom_charge(
    mol: Chem.Mol,
    atom_indices: tuple[int, ...],
    centre_idx: int,
    charge: float,
) -> FragmentLibrary:
    from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder
    from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
    pattern, g2l = SmartsBuilder.subgraph(mol, atom_indices)
    centre_local = g2l[centre_idx]
    fragment = Fragment(
        pattern=pattern,
        annotated_atoms=(AnnotatedAtom(local_index=centre_local, parameter=AtomParameter.CHARGE),),
    )
    hit = ParameterHit(
        value=charge,
        fragment=fragment,
        match_instance=0,
        member_local_indices=(centre_local,),
    )
    record = ParameterRecord(
        global_indices=(centre_idx,),
        parameter=AtomParameter.CHARGE,
        hits=(hit,),
    )
    metadata = {
        pattern: {
            centre_local: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)
        }
    }
    return FragmentLibrary(records=(record,), atom_metadata=metadata)


@pytest.fixture
def propane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    return mol


@pytest.fixture
def propane_structure(propane_mol) -> pmd.Structure:
    return TopologyBuilder.build(propane_mol)


# ---------------------------------------------------------------------------
# Tests: assign atom parameters
# ---------------------------------------------------------------------------

class TestTilerAtomAssignment:
    def test_assigns_charge_from_library(self, propane_mol, propane_structure):
        # arrange
        heavy = tuple(i for i in range(propane_mol.GetNumAtoms()) if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        centre = heavy[0]
        library = _make_library_with_atom_charge(propane_mol, heavy, centre, -0.15)
        tiler = PolymerParameterisationTiler(library=library, missing_strategies={AtomParameter: StrictMissingParameterStrategy()})

        # act / assert: at least the centre atom gets its charge; others raise with strict
        with pytest.raises(MissingParameterError):
            tiler.tile(propane_structure, propane_mol)

    def test_assigns_charge_correctly_when_library_covers_all_atoms(self, propane_mol):
        # arrange — build library with CHARGE for ALL heavy atoms
        heavy = tuple(i for i in range(propane_mol.GetNumAtoms()) if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder
        from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
        pattern, g2l = SmartsBuilder.subgraph(propane_mol, heavy)
        records = []
        metadata = {pattern: {}}
        for h_idx in heavy:
            local = g2l[h_idx]
            frag = Fragment(
                pattern=pattern,
                annotated_atoms=tuple(AnnotatedAtom(local_index=g2l[h], parameter=p) for h in heavy for p in AtomParameter),
            )
            for param in AtomParameter:
                records.append(ParameterRecord(
                    global_indices=(h_idx,),
                    parameter=param,
                    hits=(ParameterHit(value=-0.10, fragment=frag, match_instance=0, member_local_indices=(local,)),),
                ))
            metadata[pattern][local] = AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)
        library = FragmentLibrary(records=tuple(records), atom_metadata=metadata)
        structure = TopologyBuilder.build(propane_mol)

        # Need strategies for H atoms too (they won't be in library)
        # Use a permissive approach: set default value for missing
        from polymer_md.parameterisation.strategies.base import MissingParameterStrategy, StrategyContext

        class ZeroStrategy:
            def resolve(self, global_indices, parameter, context):
                return 0.0

        zero = ZeroStrategy()
        tiler = PolymerParameterisationTiler(
            library=library,
            missing_strategies={
                AtomParameter: zero,
                BondParameter: zero,
                AngleParameter: zero,
                DihedralParameter: zero,
            },
        )

        # act
        tiler.tile(structure, propane_mol)

        # assert — heavy atoms should have charge -0.10, H atoms should have charge 0.0
        for atom in structure.atoms:
            rdkit_atom = propane_mol.GetAtomWithIdx(atom.idx)
            if rdkit_atom.GetAtomicNum() != 1:
                assert pytest.approx(atom.charge, abs=1e-6) == -0.10

    def test_sets_gaff2_type_from_atom_metadata(self, propane_mol):
        # arrange
        heavy = tuple(i for i in range(propane_mol.GetNumAtoms()) if propane_mol.GetAtomWithIdx(i).GetAtomicNum() != 1)
        centre = heavy[0]
        library = _make_library_with_atom_charge(propane_mol, heavy, centre, -0.15)
        structure = TopologyBuilder.build(propane_mol)

        class ZeroStrategy:
            def resolve(self, global_indices, parameter, context):
                return 0.0

        zero = ZeroStrategy()
        tiler = PolymerParameterisationTiler(
            library=library,
            missing_strategies={
                AtomParameter: zero,
                BondParameter: zero,
                AngleParameter: zero,
                DihedralParameter: zero,
            },
        )

        # act
        tiler.tile(structure, propane_mol)

        # assert: the centre atom should have GAFF2 type "c3"
        assert structure.atoms[centre].type == "c3"


# ---------------------------------------------------------------------------
# Tests: canonical key helper
# ---------------------------------------------------------------------------

class TestCanonicalKey:
    def test_atom_key_unchanged(self):
        member = AnnotatedAtom(local_index=3, parameter=AtomParameter.CHARGE)
        key = PolymerParameterisationTiler._canonical_key((5,), member)
        assert key == (5,)

    def test_bond_key_sorted(self):
        member = AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT)
        key = PolymerParameterisationTiler._canonical_key((7, 3), member)
        assert key == (3, 7)

    def test_angle_key_canonical(self):
        member = AnnotatedAngle(local_indices=(0, 1, 2), parameter=AngleParameter.FORCE_CONSTANT)
        key = PolymerParameterisationTiler._canonical_key((8, 5, 3), member)
        assert key == (3, 5, 8)

    def test_dihedral_key_canonical(self):
        member = AnnotatedDihedral(local_indices=(0, 1, 2, 3), parameter=DihedralParameter.FORCE_CONSTANT)
        key_a = PolymerParameterisationTiler._canonical_key((1, 2, 3, 4), member)
        key_b = PolymerParameterisationTiler._canonical_key((4, 3, 2, 1), member)
        assert key_a == key_b


# ---------------------------------------------------------------------------
# Tests: member global/local indices helpers
# ---------------------------------------------------------------------------

class TestMemberIndicesHelpers:
    def test_atom_global_indices(self):
        member = AnnotatedAtom(local_index=2, parameter=AtomParameter.CHARGE)
        rdkit_match = (10, 20, 30)
        assert PolymerParameterisationTiler._member_global_indices(member, rdkit_match) == (30,)

    def test_bond_global_indices(self):
        member = AnnotatedBond(local_indices=(0, 1), parameter=BondParameter.FORCE_CONSTANT)
        rdkit_match = (10, 20)
        assert PolymerParameterisationTiler._member_global_indices(member, rdkit_match) == (10, 20)

    def test_atom_local_indices(self):
        member = AnnotatedAtom(local_index=5, parameter=AtomParameter.CHARGE)
        assert PolymerParameterisationTiler._member_local_indices(member) == (5,)

    def test_bond_local_indices(self):
        member = AnnotatedBond(local_indices=(1, 2), parameter=BondParameter.FORCE_CONSTANT)
        assert PolymerParameterisationTiler._member_local_indices(member) == (1, 2)


# ---------------------------------------------------------------------------
# Tests: set_atom_param helper
# ---------------------------------------------------------------------------

class TestSetAtomParam:
    def test_charge(self):
        atom = pmd.Atom()
        PolymerParameterisationTiler._set_atom_param(atom, AtomParameter.CHARGE, 0.5)
        assert atom.charge == pytest.approx(0.5)

    def test_epsilon(self):
        atom = pmd.Atom()
        PolymerParameterisationTiler._set_atom_param(atom, AtomParameter.EPSILON, 1.2)
        assert atom.epsilon == pytest.approx(1.2)

    def test_sigma(self):
        atom = pmd.Atom()
        PolymerParameterisationTiler._set_atom_param(atom, AtomParameter.SIGMA, 3.4)
        assert atom.sigma == pytest.approx(3.4)

    def test_mass(self):
        atom = pmd.Atom()
        PolymerParameterisationTiler._set_atom_param(atom, AtomParameter.MASS, 12.0)
        assert atom.mass == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Tests: invalid SMARTS and improper dihedral coverage
# ---------------------------------------------------------------------------

class TestCollectAssignmentsInvalidSmarts:
    def test_invalid_pattern_in_library_is_skipped(self):
        bad_fragment = Fragment(
            pattern="[invalid!!!",
            annotated_atoms=(AnnotatedAtom(local_index=0, parameter=AtomParameter.CHARGE),),
        )
        hit = ParameterHit(
            value=0.1,
            fragment=bad_fragment,
            match_instance=0,
            member_local_indices=(0,),
        )
        record = ParameterRecord(
            global_indices=(0,),
            parameter=AtomParameter.CHARGE,
            hits=(hit,),
        )
        library = FragmentLibrary(records=(record,), atom_metadata={})
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        tiler = PolymerParameterisationTiler(library=library)
        assignments = tiler._collect_assignments(mol)
        assert assignments == {}

    def test_invalid_pattern_in_atom_metadata_is_skipped(self):
        library = FragmentLibrary(
            records=(),
            atom_metadata={
                "[invalid!!!": {0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0)}
            },
        )
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        tiler = PolymerParameterisationTiler(library=library)
        gaff2_types = tiler._collect_gaff2_types(mol)
        assert gaff2_types == {}


class TestApplyDihedralsImproper:
    def test_improper_dihedral_is_skipped(self):
        structure = pmd.Structure()
        atoms = []
        for _ in range(4):
            atom = pmd.Atom()
            structure.add_atom(atom, "MOL", 1)
            atoms.append(atom)
        structure.dihedrals.append(
            pmd.Dihedral(atoms[0], atoms[1], atoms[2], atoms[3], improper=True)
        )
        library = FragmentLibrary(records=(), atom_metadata={})
        context = StrategyContext(
            library=library,
            derived_mol=Chem.MolFromSmiles("CCCC"),
            polymer_atom_metadata={},
        )
        tiler = PolymerParameterisationTiler(
            library=library,
            missing_strategies={
                DihedralParameter: type("Z", (), {"resolve": lambda self, gi, p, c: 0.0})()
            },
        )
        tiler._apply_dihedrals(structure, {}, context)
        assert structure.dihedrals[0].type is None


# ---------------------------------------------------------------------------
# Tests: charge neutrality adjustment
# ---------------------------------------------------------------------------

class TestAdjustChargeNeutrality:
    def test_neutral_structure_unchanged(self):
        structure = pmd.Structure()
        for charge in [0.5, -0.5]:
            atom = pmd.Atom()
            atom.charge = charge
            structure.add_atom(atom, "MOL", 1)
        adjust_charge_neutrality(structure)
        assert abs(sum(a.charge for a in structure.atoms)) < 1e-10

    def test_positive_excess_redistributed(self):
        structure = pmd.Structure()
        for charge in [0.1, 0.1, 0.1]:
            atom = pmd.Atom()
            atom.charge = charge
            structure.add_atom(atom, "MOL", 1)
        adjust_charge_neutrality(structure)
        assert abs(sum(a.charge for a in structure.atoms)) < 1e-10

    def test_no_change_when_already_neutral(self):
        structure = pmd.Structure()
        for charge in [0.0, 0.0]:
            atom = pmd.Atom()
            atom.charge = charge
            structure.add_atom(atom, "MOL", 1)
        before = [a.charge for a in structure.atoms]
        adjust_charge_neutrality(structure)
        after = [a.charge for a in structure.atoms]
        assert before == after
