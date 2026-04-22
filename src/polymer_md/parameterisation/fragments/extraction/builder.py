from __future__ import annotations

import logging
from dataclasses import dataclass, field

from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.match import ParameterRecord
from polymer_md.parameterisation.fragments.extraction.interior import InteriorFragmentExtractor
from polymer_md.parameterisation.fragments.extraction.terminal import TerminalFragmentExtractor
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.fragments.matching.matcher import FragmentMatcher
from polymer_md.utils.parmed_helper import StructureMolDeriver

logger = logging.getLogger(__name__)


@dataclass
class FragmentLibraryBuilder:
    interior_extractor: InteriorFragmentExtractor = field(
        default_factory=InteriorFragmentExtractor
    )
    terminal_extractor: TerminalFragmentExtractor = field(
        default_factory=TerminalFragmentExtractor
    )

    def build(self, parameterised_trimers: list[ParameterisedTrimer]) -> FragmentLibrary:
        logger.info("Building fragment library from %d trimers.", len(parameterised_trimers))
        all_records = []
        for parameterised_trimer in parameterised_trimers:
            trimer_records = self._records_for_trimer(parameterised_trimer)
            logger.info(
                "  %s: %d records extracted.",
                parameterised_trimer.trimer_result.label,
                len(trimer_records),
            )
            all_records.extend(trimer_records)
        logger.info("Fragment library built: %d total records.", len(all_records))
        return FragmentLibrary(records=tuple(all_records))

    def _records_for_trimer(
        self,
        parameterised_trimer: ParameterisedTrimer,
    ) -> list[ParameterRecord]:
        fragments = self._extract_all_fragments(parameterised_trimer)
        derived_mol = StructureMolDeriver.derive(parameterised_trimer.structure)
        matcher = FragmentMatcher(fragments)
        matches = matcher.match_all(derived_mol)
        return matcher.build_records(matches, parameterised_trimer.structure)

    def _extract_all_fragments(self, parameterised_trimer: ParameterisedTrimer):
        return (
            self.interior_extractor.extract(parameterised_trimer)
            + self.terminal_extractor.extract(parameterised_trimer)
        )
