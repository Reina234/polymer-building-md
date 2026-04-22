from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterHit, ParameterRecord
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, BondParameter
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.strategies.base import StrategyContext


# ---------------------------------------------------------------------------
# Molecule fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ethylbenzene_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCc1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    Chem.SanitizeMol(mol)
    return mol


@pytest.fixture
def propane_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    Chem.SanitizeMol(mol)
    return mol


# ---------------------------------------------------------------------------
# Library fixtures
# ---------------------------------------------------------------------------

# SMARTS for a simple aliphatic C–C bond (ethane-like fragment used in tests)
_PATTERN_CC = "[#6;A]-[#6;A]"
_PATTERN_CCC = "[#6;A]-[#6;A]-[#6;A]"


@pytest.fixture
def minimal_atom_metadata() -> dict[str, dict[int, AtomMetadata]]:
    return {
        _PATTERN_CC: {
            0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0),
            1: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=1),
        },
        _PATTERN_CCC: {
            0: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=0),
            1: AtomMetadata(gaff2_type="c3", residue_id="S", within_residue_position=1),
            2: AtomMetadata(gaff2_type="c3", residue_id="T", within_residue_position=0),
        },
    }


def _make_atom_record(
    global_idx: int,
    parameter: AtomParameter,
    value: float,
    pattern: str,
    local_idx: int,
) -> ParameterRecord:
    fragment = Fragment(pattern=pattern)
    hit = ParameterHit(
        value=value,
        fragment=fragment,
        match_instance=0,
        member_local_indices=(local_idx,),
    )
    return ParameterRecord(
        global_indices=(global_idx,),
        parameter=parameter,
        hits=(hit,),
    )


@pytest.fixture
def library_with_metadata(minimal_atom_metadata) -> FragmentLibrary:
    records = (
        # Atom 0 in _PATTERN_CC: residue S, position 0, charge -0.10
        _make_atom_record(0, AtomParameter.CHARGE, -0.10, _PATTERN_CC, 0),
        # Atom 1 in _PATTERN_CC: residue S, position 1, charge -0.12
        _make_atom_record(1, AtomParameter.CHARGE, -0.12, _PATTERN_CC, 1),
        # Atom 0 in _PATTERN_CCC: residue S, position 0, charge -0.11 (second trimer context)
        _make_atom_record(5, AtomParameter.CHARGE, -0.11, _PATTERN_CCC, 0),
        # Atom 1 in _PATTERN_CCC: residue S, position 1, charge -0.13
        _make_atom_record(6, AtomParameter.CHARGE, -0.13, _PATTERN_CCC, 1),
        # Atom 2 in _PATTERN_CCC: residue T, position 0, charge +0.05
        _make_atom_record(7, AtomParameter.CHARGE, 0.05, _PATTERN_CCC, 2),
        # A bond record (not an atom record)
        ParameterRecord(
            global_indices=(0, 1),
            parameter=BondParameter.FORCE_CONSTANT,
            hits=(
                ParameterHit(
                    value=310.0,
                    fragment=Fragment(pattern=_PATTERN_CC),
                    match_instance=0,
                    member_local_indices=(0, 1),
                ),
            ),
        ),
    )
    return FragmentLibrary(records=records, atom_metadata=minimal_atom_metadata)


@pytest.fixture
def strategy_context_s_position0(library_with_metadata, propane_mol) -> StrategyContext:
    # polymer_atom_metadata: atom 0 is residue S, position 0
    return StrategyContext(
        library=library_with_metadata,
        derived_mol=propane_mol,
        polymer_atom_metadata={0: ("S", 0)},
    )


@pytest.fixture
def strategy_context_s_position1(library_with_metadata, propane_mol) -> StrategyContext:
    return StrategyContext(
        library=library_with_metadata,
        derived_mol=propane_mol,
        polymer_atom_metadata={1: ("S", 1)},
    )
