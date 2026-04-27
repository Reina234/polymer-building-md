from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from polymer_md.parameterisation.fragments.data_models.annotated_members import (
    AnnotatedAngle,
    AnnotatedAtom,
    AnnotatedBond,
    AnnotatedDihedral,
    AnnotatedImproper,
    AnnotatedMember,
)
from polymer_md.parameterisation.fragments.data_models.atom_metadata import AtomMetadata
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
    DihedralTerm,
    ForceFieldParameter,
    ImproperParameter,
)


class ParameterKind(StrEnum):
    BOND = "bond"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    IMPROPER_DIHEDRAL = "improper_dihedral"
    ATOM = "atom"


_KIND_TO_CLASS: dict[ParameterKind, type] = {
    ParameterKind.BOND: BondParameter,
    ParameterKind.ANGLE: AngleParameter,
    ParameterKind.DIHEDRAL: DihedralParameter,
    ParameterKind.IMPROPER_DIHEDRAL: ImproperParameter,
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


class _MetaField(StrEnum):
    GAFF2_TYPE = "gaff2_type"
    RESIDUE_ID = "residue_id"
    WITHIN_RESIDUE_POSITION = "within_residue_position"


_SCHEMA_VERSION = "2"
_SUPPORTED_VERSIONS = {"2"}


@dataclass(frozen=True)
class FragmentLibrary:
    records: tuple[ParameterRecord, ...]
    atom_metadata: dict[str, dict[int, AtomMetadata]] = field(default_factory=dict)
    schema_version: str = field(default=_SCHEMA_VERSION)

    def query(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
    ) -> ParameterRecord | None:
        for record in self.records:
            if record.global_indices == global_indices and record.parameter == parameter:
                return record
        return None

    def hit_values_for_residue_position(
        self,
        residue_id: str,
        within_residue_position: int,
        parameter: ForceFieldParameter,
    ) -> list[float]:
        targets = self._targets_for_residue_position(residue_id, within_residue_position)
        values = []
        for record in self.records:
            if record.parameter != parameter:
                continue
            for hit in record.hits:
                if len(hit.member_local_indices) != 1:
                    continue
                if (hit.fragment.pattern, hit.member_local_indices[0]) in targets:
                    values.append(hit.value)
        return values

    def metadata_for(self, pattern: str, local_idx: int) -> AtomMetadata | None:
        return self.atom_metadata.get(pattern, {}).get(local_idx)

    def values_for_member(
        self,
        pattern: str,
        member_local_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
    ) -> list[float | tuple[DihedralTerm, ...]]:
        values: list[float | tuple[DihedralTerm, ...]] = []
        for record in self.records:
            if record.parameter != parameter:
                continue
            for hit in record.hits:
                if hit.fragment.pattern == pattern and hit.member_local_indices == member_local_indices:
                    values.append(hit.value)
        return values

    def _targets_for_residue_position(
        self,
        residue_id: str,
        within_residue_position: int,
    ) -> set[tuple[str, int]]:
        targets: set[tuple[str, int]] = set()
        for pattern, local_map in self.atom_metadata.items():
            for local_idx, meta in local_map.items():
                if meta.residue_id == residue_id and meta.within_residue_position == within_residue_position:
                    targets.add((pattern, local_idx))
        return targets

    @classmethod
    def merge(cls, first: FragmentLibrary, second: FragmentLibrary) -> FragmentLibrary:
        if first.schema_version != second.schema_version:
            raise ValueError(
                f"Cannot merge libraries with different schema versions: "
                f"{first.schema_version!r} vs {second.schema_version!r}"
            )
        return cls(
            records=cls._merged_records(first.records, second.records),
            atom_metadata={**first.atom_metadata, **second.atom_metadata},
            schema_version=first.schema_version,
        )

    @staticmethod
    def _merged_records(
        first: tuple[ParameterRecord, ...],
        second: tuple[ParameterRecord, ...],
    ) -> tuple[ParameterRecord, ...]:
        seen = {(r.global_indices, r.parameter) for r in first}
        additional = [r for r in second if (r.global_indices, r.parameter) not in seen]
        return first + tuple(additional)

    def save(self, path: Path) -> None:
        data = {
            "schema_version": self.schema_version,
            "records": [self._record_to_dict(record) for record in self.records],
            "atom_metadata": self._metadata_to_dict(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> FragmentLibrary:
        data = json.loads(path.read_text())
        version = data.get("schema_version", "1")
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported FragmentLibrary schema version {version!r}. "
                f"Supported: {sorted(_SUPPORTED_VERSIONS)}. "
                f"Run migrate_v1_to_v2() to upgrade."
            )
        records = tuple(cls._record_from_dict(entry) for entry in data["records"])
        atom_metadata = cls._metadata_from_dict(data.get("atom_metadata", {}))
        return cls(records=records, atom_metadata=atom_metadata, schema_version=version)

    def _metadata_to_dict(self) -> dict:
        result: dict = {}
        for pattern, local_map in self.atom_metadata.items():
            result[pattern] = {}
            for local_idx, meta in local_map.items():
                result[pattern][str(local_idx)] = {
                    _MetaField.GAFF2_TYPE: meta.gaff2_type,
                    _MetaField.RESIDUE_ID: meta.residue_id,
                    _MetaField.WITHIN_RESIDUE_POSITION: meta.within_residue_position,
                }
        return result

    @staticmethod
    def _metadata_from_dict(data: dict) -> dict[str, dict[int, AtomMetadata]]:
        result: dict[str, dict[int, AtomMetadata]] = {}
        for pattern, local_map in data.items():
            result[pattern] = {}
            for local_idx, meta in local_map.items():
                result[pattern][int(local_idx)] = AtomMetadata(
                    gaff2_type=meta[_MetaField.GAFF2_TYPE],
                    residue_id=meta[_MetaField.RESIDUE_ID],
                    within_residue_position=meta[_MetaField.WITHIN_RESIDUE_POSITION],
                )
        return result

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
        if isinstance(hit.value, tuple):
            serialised_value = {"terms": [[t.force_constant, t.phase, t.periodicity] for t in hit.value]}
        else:
            serialised_value = hit.value
        return {
            _HitField.VALUE: serialised_value,
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
        raw = data[_HitField.VALUE]
        if isinstance(raw, dict) and "terms" in raw:
            value: float | tuple[DihedralTerm, ...] = tuple(
                DihedralTerm(force_constant=t[0], phase=t[1], periodicity=t[2])
                for t in raw["terms"]
            )
        else:
            value = raw
        return ParameterHit(
            value=value,
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
            annotated_impropers=tuple(m for m in members if isinstance(m, AnnotatedImproper)),
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
        if isinstance(parameter, ImproperParameter):
            return AnnotatedImproper(local_indices=local_indices, parameter=parameter)
        return AnnotatedAtom(local_index=local_indices[0], parameter=parameter)

    @staticmethod
    def _deserialise_parameter(kind: str, name: str) -> ForceFieldParameter:
        parameter_class = _KIND_TO_CLASS[ParameterKind(kind)]
        return parameter_class[name]
