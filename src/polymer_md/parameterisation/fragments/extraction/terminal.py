from __future__ import annotations

import parmed as pmd
from dataclasses import dataclass

from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.extraction.region_extractor import RegionFragmentExtractor
from polymer_md.utils.parmed_helper import CoordinateCrosswalk, StructureMolDeriver


@dataclass
class TerminalFragmentExtractor:
    def extract(self, parameterised_trimer: ParameterisedTrimer) -> list[tuple[Fragment, dict[int, int]]]:
        derived_mol = StructureMolDeriver.derive(parameterised_trimer.structure)
        mol3d_to_parmed = CoordinateCrosswalk.map_mol3d_to_parmed(
            parameterised_trimer.mol_3d,
            parameterised_trimer.structure,
        )
        trimer_result = parameterised_trimer.trimer_result
        extractor = RegionFragmentExtractor()

        left_indices = extractor.resolve_parmed_indices(
            heavy_atom_indices=trimer_result.left_atom_indices,
            mol_3d=parameterised_trimer.mol_3d,
            mol3d_to_parmed=mol3d_to_parmed,
        )
        central_indices = extractor.resolve_parmed_indices(
            heavy_atom_indices=trimer_result.central_atom_indices,
            mol_3d=parameterised_trimer.mol_3d,
            mol3d_to_parmed=mol3d_to_parmed,
        )
        right_indices = extractor.resolve_parmed_indices(
            heavy_atom_indices=trimer_result.right_atom_indices,
            mol_3d=parameterised_trimer.mol_3d,
            mol3d_to_parmed=mol3d_to_parmed,
        )
        cap_indices = extractor.resolve_parmed_indices(
            heavy_atom_indices=trimer_result.cap_atom_indices,
            mol_3d=parameterised_trimer.mol_3d,
            mol3d_to_parmed=mol3d_to_parmed,
        )

        left_cap, right_cap = self._split_caps_by_adjacency(
            cap_indices, left_indices, right_indices, parameterised_trimer.structure
        )

        results = [
            extractor.extract(
                derived_mol=derived_mol,
                structure=parameterised_trimer.structure,
                context_parmed_indices=left_indices | central_indices | left_cap,
                region_parmed_indices=left_indices,
            ),
            extractor.extract(
                derived_mol=derived_mol,
                structure=parameterised_trimer.structure,
                context_parmed_indices=right_indices | central_indices | right_cap,
                region_parmed_indices=right_indices,
            ),
        ]

        if left_cap:
            results.append(
                extractor.extract(
                    derived_mol=derived_mol,
                    structure=parameterised_trimer.structure,
                    context_parmed_indices=left_cap | left_indices,
                    region_parmed_indices=left_cap,
                )
            )
        if right_cap:
            results.append(
                extractor.extract(
                    derived_mol=derived_mol,
                    structure=parameterised_trimer.structure,
                    context_parmed_indices=right_cap | right_indices,
                    region_parmed_indices=right_cap,
                )
            )

        return results

    @staticmethod
    def _split_caps_by_adjacency(
        cap_indices: frozenset[int],
        left_indices: frozenset[int],
        right_indices: frozenset[int],
        structure: pmd.Structure,
    ) -> tuple[frozenset[int], frozenset[int]]:
        if not cap_indices:
            return frozenset(), frozenset()
        left_anchors: set[int] = set()
        right_anchors: set[int] = set()
        for bond in structure.bonds:
            i, j = bond.atom1.idx, bond.atom2.idx
            if i in cap_indices and j in left_indices:
                left_anchors.add(i)
            elif j in cap_indices and i in left_indices:
                left_anchors.add(j)
            elif i in cap_indices and j in right_indices:
                right_anchors.add(i)
            elif j in cap_indices and i in right_indices:
                right_anchors.add(j)

        left_cap = TerminalFragmentExtractor._expand_within_cap(
            left_anchors, cap_indices, structure
        )
        right_cap = TerminalFragmentExtractor._expand_within_cap(
            right_anchors, cap_indices, structure
        )
        return frozenset(left_cap), frozenset(right_cap)

    @staticmethod
    def _expand_within_cap(
        anchors: set[int],
        cap_indices: frozenset[int],
        structure: pmd.Structure,
    ) -> set[int]:
        visited: set[int] = set(anchors)
        frontier = list(anchors)
        while frontier:
            current = frontier.pop()
            for bond in structure.bonds:
                i, j = bond.atom1.idx, bond.atom2.idx
                if i == current and j in cap_indices and j not in visited:
                    visited.add(j)
                    frontier.append(j)
                elif j == current and i in cap_indices and i not in visited:
                    visited.add(i)
                    frontier.append(i)
        return visited
