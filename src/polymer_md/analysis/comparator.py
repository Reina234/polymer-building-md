from __future__ import annotations

import logging
from dataclasses import dataclass

import parmed as pmd
from rdkit import Chem

logger = logging.getLogger(__name__)

from polymer_md.analysis.results import AnalysisResult, ComparisonResult
from polymer_md.parameterisation.data_models.parameterised_mol import ParameterisedMolecule
from polymer_md.parameterisation.fragments.data_models.annotated_members import AnnotatedAtom
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.matching.matcher import FragmentMatcher


@dataclass
class ParameterComparator:
    fragments: list[Fragment]

    def extract(self, structure: pmd.Structure, mol: Chem.Mol) -> AnalysisResult:
        derived_mol = mol
        result: AnalysisResult = {}
        for fragment in self.fragments:
            result[fragment] = self._extract_fragment(fragment, structure, derived_mol)
        return result

    def compare(self, molecules: dict[str, ParameterisedMolecule]) -> ComparisonResult:
        results = {}
        for label, parameterised_mol in molecules.items():
            results[label] = self.extract(parameterised_mol.structure, parameterised_mol.mol)
        return ComparisonResult(results=results)

    @staticmethod
    def _extract_fragment(
        fragment: Fragment,
        structure: pmd.Structure,
        derived_mol: Chem.Mol,
    ) -> dict:
        query = Chem.MolFromSmarts(fragment.pattern)
        if query is None:
            return {}
        matches = derived_mol.GetSubstructMatches(query)
        param_values: dict = {}
        for member in fragment.all_members:
            values = []
            for rdkit_match in matches:
                global_indices = ParameterComparator._member_global_indices(member, rdkit_match)
                try:
                    value = FragmentMatcher._extract_value(structure, global_indices, member)
                    values.append(value)
                except (ValueError, AttributeError, TypeError) as error:
                    logger.warning(
                        "Parameter lookup failed for %s indices %s: %s",
                        type(member.parameter).__name__,
                        global_indices,
                        error,
                    )
            if values:
                param_values[member.parameter] = values
        return param_values

    @staticmethod
    def _member_global_indices(
        member,
        rdkit_match: tuple[int, ...],
    ) -> tuple[int, ...]:
        if isinstance(member, AnnotatedAtom):
            return (rdkit_match[member.local_index],)
        return tuple(rdkit_match[i] for i in member.local_indices)
