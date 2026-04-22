from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
    AnnotatedMember,
)
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.data_models.match import (
    ParameterHit,
    ParameterRecord,
)
from polymer_md.parameterisation.fragments.data_models.parameters import (
    AngleParameter,
    AtomParameter,
    BondParameter,
    DihedralParameter,
    ForceFieldParameter,
)


class ParameterKind(StrEnum):
    BOND = "bond"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    ATOM = "atom"


_KIND_TO_CLASS: dict[ParameterKind, type] = {
    ParameterKind.BOND: BondParameter,
    ParameterKind.ANGLE: AngleParameter,
    ParameterKind.DIHEDRAL: DihedralParameter,
    ParameterKind.ATOM: AtomParameter,
}
_CLASS_TO_KIND: dict[type, ParameterKind] = {cls: kind for kind, cls in _KIND_TO_CLASS.items()}


class _RecordField(StrEnum):
    GLOBAL_INDICES = "global_indices"
    PARAMETER_KIND = "parameter_kind"
    PARAMETER_NAME = "parameter_name"
    HITS = "hits"


class _HitField(StrEnum):
    VALUE = "value"
    FRAGMENT_PATTERN = "fragment_pattern"
    MATCH_INSTANCE = "match_instance"
    MEMBER_LOCAL_INDICES = "member_local_indices"


@dataclass(frozen=True)
class FragmentLibrary:
    records: tuple[ParameterRecord, ...]

    def query(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
    ) -> ParameterRecord | None:
        for record in self.records:
            if record.global_indices == global_indices and record.parameter == parameter:
                return record
        return None

    def save(self, path: Path) -> None:
        data = [self._record_to_dict(record) for record in self.records]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> FragmentLibrary:
        data = json.loads(path.read_text())
        return cls(records=tuple(cls._record_from_dict(entry) for entry in data))

    @staticmethod
    def _record_to_dict(record: ParameterRecord) -> dict:
        return {
            _RecordField.GLOBAL_INDICES: list(record.global_indices),
            _RecordField.PARAMETER_KIND: _CLASS_TO_KIND[type(record.parameter)],
            _RecordField.PARAMETER_NAME: record.parameter.name,
            _RecordField.HITS: [FragmentLibrary._hit_to_dict(hit) for hit in record.hits],
        }

    @staticmethod
    def _hit_to_dict(hit: ParameterHit) -> dict:
        return {
            _HitField.VALUE: hit.value,
            _HitField.FRAGMENT_PATTERN: hit.fragment.pattern,
            _HitField.MATCH_INSTANCE: hit.match_instance,
            _HitField.MEMBER_LOCAL_INDICES: list(hit.member_local_indices),
        }

    @staticmethod
    def _record_from_dict(data: dict) -> ParameterRecord:
        parameter = FragmentLibrary._deserialise_parameter(
            data[_RecordField.PARAMETER_KIND], data[_RecordField.PARAMETER_NAME]
        )
        hits = tuple(FragmentLibrary._hit_from_dict(hit_data) for hit_data in data[_RecordField.HITS])
        return ParameterRecord(
            global_indices=tuple(data[_RecordField.GLOBAL_INDICES]),
            parameter=parameter,
            hits=hits,
        )

    @staticmethod
    def _hit_from_dict(data: dict) -> ParameterHit:
        return ParameterHit(
            value=data[_HitField.VALUE],
            fragment=Fragment(pattern=data[_HitField.FRAGMENT_PATTERN]),
            match_instance=data[_HitField.MATCH_INSTANCE],
            member_local_indices=tuple(data[_HitField.MEMBER_LOCAL_INDICES]),
        )

    def to_fragments(self) -> list[Fragment]:
        records_by_pattern = self._group_records_by_pattern()
        return [
            self._reconstruct_fragment(pattern, records)
            for pattern, records in records_by_pattern.items()
        ]

    def _group_records_by_pattern(self) -> dict[str, list[ParameterRecord]]:
        groups: dict[str, list[ParameterRecord]] = {}
        for record in self.records:
            pattern = self._pattern_for_record(record)
            if pattern is None:
                continue
            groups.setdefault(pattern, []).append(record)
        return groups

    @staticmethod
    def _pattern_for_record(record: ParameterRecord) -> str | None:
        if not record.hits:
            return None
        return record.hits[0].fragment.pattern

    def _reconstruct_fragment(
        self,
        pattern: str,
        records: list[ParameterRecord],
    ) -> Fragment:
        members = self._build_annotated_members(records)
        return Fragment(
            pattern=pattern,
            annotated_bonds=tuple(m for m in members if isinstance(m, AnnotatedBond)),
            annotated_angles=tuple(m for m in members if isinstance(m, AnnotatedAngle)),
            annotated_dihedrals=tuple(m for m in members if isinstance(m, AnnotatedDihedral)),
            annotated_atoms=tuple(m for m in members if isinstance(m, AnnotatedAtom)),
        )

    @staticmethod
    def _build_annotated_members(records: list[ParameterRecord]) -> list[AnnotatedMember]:
        seen: set[tuple] = set()
        members = []
        for record in records:
            if not record.hits:
                continue
            local_indices = record.hits[0].member_local_indices
            key = (record.parameter, local_indices)
            if key in seen:
                continue
            seen.add(key)
            members.append(FragmentLibrary._build_member(record.parameter, local_indices))
        return members

    @staticmethod
    def _build_member(
        parameter: ForceFieldParameter,
        local_indices: tuple[int, ...],
    ) -> AnnotatedMember:
        if isinstance(parameter, BondParameter):
            return AnnotatedBond(local_indices=local_indices, parameter=parameter)
        if isinstance(parameter, AngleParameter):
            return AnnotatedAngle(local_indices=local_indices, parameter=parameter)
        if isinstance(parameter, DihedralParameter):
            return AnnotatedDihedral(local_indices=local_indices, parameter=parameter)
        return AnnotatedAtom(local_index=local_indices[0], parameter=parameter)

    @staticmethod
    def _deserialise_parameter(kind: str, name: str) -> ForceFieldParameter:
        parameter_class = _KIND_TO_CLASS[ParameterKind(kind)]
        return parameter_class[name]
