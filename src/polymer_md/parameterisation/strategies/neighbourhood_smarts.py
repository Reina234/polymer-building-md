from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
from polymer_md.parameterisation.fragments.data_models.match import ParameterRecord
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter, ForceFieldParameter
from polymer_md.parameterisation.fragments.extraction.smarts_builder import SmartsBuilder
from polymer_md.parameterisation.strategies.base import MissingParameterError, StrategyContext


@dataclass
class NeighbourhoodSMARTSStrategy:
    max_radius: int = 3
    min_radius: int = 1
    min_matches: int = 1

    def resolve(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> float:
        if not isinstance(parameter, AtomParameter):
            raise MissingParameterError(
                f"NeighbourhoodSMARTSStrategy only supports AtomParameter, got {type(parameter).__name__}"
            )

        atom_idx = global_indices[0]
        residue_id, within_residue_position = self._polymer_residue_position(
            atom_idx, context
        )

        for radius in range(self.max_radius, self.min_radius - 1, -1):
            values = self._collect_values_at_radius(
                atom_idx, radius, residue_id, within_residue_position, parameter, context
            )
            if len(values) >= self.min_matches:
                return sum(values) / len(values)

        raise MissingParameterError(
            f"NeighbourhoodSMARTSStrategy: no library matches found for atom {atom_idx} "
            f"({residue_id}, position={within_residue_position}, {parameter}) "
            f"at any radius {self.min_radius}–{self.max_radius} with min_matches={self.min_matches}."
        )

    def _collect_values_at_radius(
        self,
        atom_idx: int,
        radius: int,
        residue_id: str,
        within_residue_position: int,
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> list[float]:
        result = self._neighbourhood_query(context.derived_mol, atom_idx, radius)
        if result is None:
            return []
        neighbourhood_query, centre_local_idx = result

        values: list[float] = []
        seen_records: set[int] = set()
        for pattern, local_map in context.library.atom_metadata.items():
            matched_locals = self._matched_local_indices(
                pattern, neighbourhood_query, centre_local_idx,
                local_map, residue_id, within_residue_position,
            )
            for local_idx in matched_locals:
                value = self._record_value(
                    pattern, local_idx, parameter, context.library.records, seen_records
                )
                if value is not None:
                    values.append(value)
        return values

    def _neighbourhood_query(
        self,
        mol: Chem.Mol,
        atom_idx: int,
        radius: int,
    ) -> tuple[Chem.Mol, int] | None:
        full_ball = self._ball_around_atom(mol, atom_idx, radius)
        heavy_ball = tuple(
            idx for idx in full_ball
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        )
        if not heavy_ball:
            return None
        smarts, centre_to_local = SmartsBuilder.subgraph(mol, heavy_ball, include_h_count=False)
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            return None
        return query, centre_to_local[atom_idx]

    @staticmethod
    def _matched_local_indices(
        pattern: str,
        neighbourhood_query: Chem.Mol,
        centre_local_idx: int,
        local_map: dict[int, AtomMetadata],
        residue_id: str,
        within_residue_position: int,
    ) -> list[int]:
        fragment_mol = Chem.MolFromSmarts(pattern)
        if fragment_mol is None:
            return []
        matched = []
        for submatch in fragment_mol.GetSubstructMatches(neighbourhood_query):
            local_idx = submatch[centre_local_idx]
            meta = local_map.get(local_idx)
            if meta is not None and meta.residue_id == residue_id and meta.within_residue_position == within_residue_position:
                matched.append(local_idx)
        return matched

    @staticmethod
    def _record_value(
        pattern: str,
        local_idx: int,
        parameter: ForceFieldParameter,
        records: tuple[ParameterRecord, ...],
        seen_records: set[int],
    ) -> float | None:
        for record_idx, record in enumerate(records):
            if record_idx in seen_records or record.parameter != parameter:
                continue
            for hit in record.hits:
                if (
                    hit.fragment.pattern == pattern
                    and len(hit.member_local_indices) == 1
                    and hit.member_local_indices[0] == local_idx
                ):
                    seen_records.add(record_idx)
                    return hit.value
        return None

    @staticmethod
    def _ball_around_atom(mol: Chem.Mol, atom_idx: int, radius: int) -> tuple[int, ...]:
        visited = {atom_idx}
        frontier = {atom_idx}
        for _ in range(radius):
            new_frontier = set()
            for current in frontier:
                for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx not in visited:
                        visited.add(nbr_idx)
                        new_frontier.add(nbr_idx)
            frontier = new_frontier
        return tuple(sorted(visited))

    @staticmethod
    def _polymer_residue_position(
        atom_idx: int,
        context: StrategyContext,
    ) -> tuple[str, int]:
        result = context.polymer_atom_metadata.get(atom_idx)
        if result is None:
            raise MissingParameterError(
                f"Atom index {atom_idx} has no residue position metadata in context. "
                f"Ensure polymer_atom_metadata is populated before calling this strategy."
            )
        return result
