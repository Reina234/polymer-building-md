from __future__ import annotations

import logging

from polymer_md.chemistry.data_models import GroupType, ReactionRule
from polymer_md.core import DuplicateRuleError, OverrideConflictError

logger = logging.getLogger(__name__)


class ChemistryRegistry:
    def __init__(self) -> None:
        self._groups: dict[str, GroupType] = {}
        self._rules: dict[frozenset[str], ReactionRule] = {}
        self._validated = False

    def add_group(self, group: GroupType) -> None:
        self._groups[group.name] = group
        self._validated = False

    def add_rule(self, rule: ReactionRule) -> None:
        if rule.groups in self._rules:
            raise DuplicateRuleError(rule.groups, self._rules[rule.groups].name)
        self._rules[rule.groups] = rule
        self._validated = False

    def validate(self) -> None:
        if self._validated:
            return
        self._check_for_mutual_overrides()
        self._warn_on_missing_override_targets()
        self._validated = True

    def lookup_rule(
        self, group_name: str, other_group_name: str
    ) -> ReactionRule | None:
        self.validate()
        key = frozenset({group_name, other_group_name})
        return self._rules.get(key)

    def partners_for(self, group_name: str) -> list[GroupType]:
        self.validate()
        partner_names = self._find_partner_names(group_name)
        return [self._groups[name] for name in partner_names if name in self._groups]

    @property
    def groups(self) -> frozenset[GroupType]:
        return frozenset(self._groups.values())

    @property
    def rules(self) -> frozenset[ReactionRule]:
        return frozenset(self._rules.values())

    def _check_for_mutual_overrides(self) -> None:
        for group in self._groups.values():
            for overridden_name in group.overrides:
                overridden = self._groups.get(overridden_name)
                if overridden is not None and group.name in overridden.overrides:
                    raise OverrideConflictError(group.name, overridden_name)

    def _warn_on_missing_override_targets(self) -> None:
        for group in self._groups.values():
            for overridden_name in group.overrides:
                if overridden_name not in self._groups:
                    logger.warning(
                        "Group '%s' declares override of unknown group '%s'. Override skipped.",
                        group.name,
                        overridden_name,
                    )

    def _find_partner_names(self, group_name: str) -> list[str]:
        partner_names: list[str] = []
        for key in self._rules:
            if group_name in key:
                for name in key:
                    if name != group_name:
                        partner_names.append(name)
        return partner_names
