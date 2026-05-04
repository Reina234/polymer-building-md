from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from polymer_md.analysis.comparator import ParameterComparator
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.matching.matcher import FragmentMatcher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FragmentDeviationSummary:
    pattern: str
    parameter_name: str
    our_mean: float
    our_std: float
    ref_mean: float
    ref_std: float
    pct_diff: float
    our_count: int
    ref_count: int


@dataclass(frozen=True)
class AtomDeviation:
    per_atom: dict[int, float]
    per_atom_details: dict[int, list[tuple[str, float, float]]]
    summaries: list[FragmentDeviationSummary]


def compute_atom_deviations(
    molecule: ParameterisedMolecule,
    reference: ParameterisedMolecule,
    fragments: list[Fragment],
) -> AtomDeviation:
    from rdkit import Chem

    atom_diffs: dict[int, list[float]] = defaultdict(list)
    atom_details: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
    summaries: list[FragmentDeviationSummary] = []

    for fragment in fragments:
        query = Chem.MolFromSmarts(fragment.pattern)
        if query is None:
            continue

        our_matches = molecule.mol.GetSubstructMatches(query)
        ref_matches = reference.mol.GetSubstructMatches(query)

        for member in fragment.all_members:
            ref_values = _extract_scalar_values(reference.structure, ref_matches, member)
            if not ref_values:
                continue

            ref_mean = float(np.mean(ref_values))
            ref_std = float(np.std(ref_values))
            our_values = _extract_scalar_values(molecule.structure, our_matches, member)

            if our_values:
                our_mean = float(np.mean(our_values))
                our_std = float(np.std(our_values))
                pct_diff = (our_mean - ref_mean) / (abs(ref_mean) + 1e-10) * 100
            else:
                our_mean = our_std = pct_diff = 0.0

            summaries.append(FragmentDeviationSummary(
                pattern=fragment.pattern,
                parameter_name=member.parameter.name,
                our_mean=our_mean,
                our_std=our_std,
                ref_mean=ref_mean,
                ref_std=ref_std,
                pct_diff=pct_diff,
                our_count=len(our_values),
                ref_count=len(ref_values),
            ))

            label = f"{fragment.pattern} / {member.parameter.name}"
            for rdkit_match in our_matches:
                global_indices = ParameterComparator._member_global_indices(member, rdkit_match)
                try:
                    our_val = float(FragmentMatcher._extract_value(
                        molecule.structure, global_indices, member
                    ))
                    pct = (our_val - ref_mean) / (abs(ref_mean) + 1e-10) * 100
                    for idx in global_indices:
                        atom_diffs[idx].append(pct)
                        atom_details[idx].append((label, our_val, ref_mean))
                except (ValueError, AttributeError, TypeError) as exc:
                    logger.debug("Deviation extraction failed for %s: %s", global_indices, exc)

    per_atom = {idx: float(np.mean(diffs)) for idx, diffs in atom_diffs.items()}
    return AtomDeviation(
        per_atom=per_atom,
        per_atom_details=dict(atom_details),
        summaries=summaries,
    )


def _extract_scalar_values(structure, matches, member) -> list[float]:
    values = []
    for rdkit_match in matches:
        global_indices = ParameterComparator._member_global_indices(member, rdkit_match)
        try:
            val = FragmentMatcher._extract_value(structure, global_indices, member)
            if isinstance(val, float):
                values.append(val)
        except (ValueError, AttributeError, TypeError):
            pass
    return values
