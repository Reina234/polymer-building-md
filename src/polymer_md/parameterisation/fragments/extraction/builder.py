from __future__ import annotations

import logging
from dataclasses import dataclass, field

import parmed as pmd

from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import ParameterRecord
from polymer_md.parameterisation.fragments.extraction.interior import InteriorFragmentExtractor
from polymer_md.parameterisation.fragments.extraction.region_extractor import RegionFragmentExtractor
from polymer_md.parameterisation.fragments.extraction.terminal import TerminalFragmentExtractor
from polymer_md.parameterisation.fragments.library import FragmentLibrary
from polymer_md.parameterisation.fragments.matching.matcher import FragmentMatcher
from polymer_md.utils.parmed_helper import CoordinateCrosswalk, StructureMolDeriver

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
        all_records: list[ParameterRecord] = []
        all_metadata: dict[str, dict[int, AtomMetadata]] = {}
        for parameterised_trimer in parameterised_trimers:
            records, metadata = self._process_trimer(parameterised_trimer)
            logger.info(
                "  %s: %d records extracted.",
                parameterised_trimer.trimer_result.label,
                len(records),
            )
            all_records.extend(records)
            all_metadata.update(metadata)
        logger.info("Fragment library built: %d total records.", len(all_records))
        return FragmentLibrary(records=tuple(all_records), atom_metadata=all_metadata)

    def _process_trimer(
        self,
        parameterised_trimer: ParameterisedTrimer,
    ) -> tuple[list[ParameterRecord], dict[str, dict[int, AtomMetadata]]]:
        fragment_pairs = self._extract_all_fragment_pairs(parameterised_trimer)
        fragments = [fragment for fragment, _ in fragment_pairs]
        derived_mol = StructureMolDeriver.derive(parameterised_trimer.structure)
        matcher = FragmentMatcher(fragments)
        matches = matcher.match_all(derived_mol)
        records = matcher.build_records(matches, parameterised_trimer.structure)
        metadata = self._build_atom_metadata(fragment_pairs, parameterised_trimer)
        return records, metadata

    def _extract_all_fragment_pairs(
        self,
        parameterised_trimer: ParameterisedTrimer,
    ) -> list[tuple[Fragment, dict[int, int]]]:
        return (
            self.interior_extractor.extract(parameterised_trimer)
            + self.terminal_extractor.extract(parameterised_trimer)
        )

    def _build_atom_metadata(
        self,
        fragment_pairs: list[tuple[Fragment, dict[int, int]]],
        parameterised_trimer: ParameterisedTrimer,
    ) -> dict[str, dict[int, AtomMetadata]]:
        mol3d_to_parmed = CoordinateCrosswalk.map_mol3d_to_parmed(
            parameterised_trimer.mol_3d,
            parameterised_trimer.structure,
        )
        parmed_to_mol3d = {v: k for k, v in mol3d_to_parmed.items()}
        region_maps = self._build_region_position_maps(parameterised_trimer, mol3d_to_parmed)

        metadata: dict[str, dict[int, AtomMetadata]] = {}
        for fragment, global_to_local in fragment_pairs:
            pattern = fragment.pattern
            if pattern in metadata:
                continue
            metadata[pattern] = self._metadata_for_pattern(
                global_to_local,
                parameterised_trimer.structure,
                parmed_to_mol3d,
                region_maps,
            )
        return metadata

    @staticmethod
    def _build_region_position_maps(
        parameterised_trimer: ParameterisedTrimer,
        mol3d_to_parmed: dict[int, int],
    ) -> dict[int, tuple[str, int]]:
        trimer_result = parameterised_trimer.trimer_result
        regions = [
            (trimer_result.left_id, trimer_result.left_atom_indices),
            (trimer_result.central_id, trimer_result.central_atom_indices),
            (trimer_result.right_id, trimer_result.right_atom_indices),
        ]
        return FragmentLibraryBuilder._region_position_entries(regions, mol3d_to_parmed)

    @staticmethod
    def _region_position_entries(
        regions: list[tuple[str, frozenset[int]]],
        mol3d_to_parmed: dict[int, int],
    ) -> dict[int, tuple[str, int]]:
        parmed_to_residue_position: dict[int, tuple[str, int]] = {}
        for residue_id, mol3d_heavy_indices in regions:
            for position, mol3d_idx in enumerate(sorted(mol3d_heavy_indices)):
                parmed_idx = mol3d_to_parmed.get(mol3d_idx)
                if parmed_idx is not None:
                    parmed_to_residue_position[parmed_idx] = (residue_id, position)
        return parmed_to_residue_position

    @staticmethod
    def _metadata_for_pattern(
        global_to_local: dict[int, int],
        structure: pmd.Structure,
        parmed_to_mol3d: dict[int, int],
        region_maps: dict[int, tuple[str, int]],
    ) -> dict[int, AtomMetadata]:
        local_metadata: dict[int, AtomMetadata] = {}
        for parmed_idx, local_idx in global_to_local.items():
            atom = structure.atoms[parmed_idx]
            if atom.atomic_number == 1:
                continue
            residue_position = region_maps.get(parmed_idx)
            if residue_position is None:
                continue
            residue_id, within_residue_position = residue_position
            gaff2_type = atom.atom_type.name if atom.atom_type is not None else ""
            local_metadata[local_idx] = AtomMetadata(
                gaff2_type=gaff2_type,
                residue_id=residue_id,
                within_residue_position=within_residue_position,
            )
        return local_metadata
