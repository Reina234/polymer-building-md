from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAtom,
    AnnotatedMember,
)
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.parameters import DihedralTerm, ForceFieldParameter

ParameterValue = float | tuple[DihedralTerm, ...]


class ResolutionStrategy(ABC):
    @abstractmethod
    def resolve(self, values: list[float]) -> float: ...


class NoMatchError(Exception):
    pass


@dataclass(frozen=True)
class FragmentMatch:
    fragment: Fragment
    atom_map: tuple[tuple[int, int], ...]  # (local_index, global_index)

    def global_indices_for(self, member: AnnotatedMember) -> tuple[int, ...]:
        local_to_global = dict(self.atom_map)
        if isinstance(member, AnnotatedAtom):
            return (local_to_global[member.local_index],)
        return tuple(local_to_global[i] for i in member.local_indices)


@dataclass(frozen=True)
class ParameterHit:
    value: ParameterValue
    fragment: Fragment
    match_instance: int
    member_local_indices: tuple[int, ...]


@dataclass(frozen=True)
class ParameterRecord:
    global_indices: tuple[int, ...]
    parameter: ForceFieldParameter
    hits: tuple[ParameterHit, ...]

    def resolve(self, strategy: ResolutionStrategy) -> float:
        if not self.hits:
            raise NoMatchError(
                f"No parameter hits for atoms {self.global_indices} "
                f"with parameter {self.parameter}"
            )
        return strategy.resolve([hit.value for hit in self.hits])  # type: ignore[arg-type]
