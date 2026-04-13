from __future__ import annotations


class RegistryError(ValueError):
    pass


class OverrideConflictError(RegistryError):
    def __init__(self, group_a: str, group_b: str) -> None:
        super().__init__(
            f"Mutual override conflict between '{group_a}' and '{group_b}'. "
            "Overrides must be directed (one side only)."
        )


class DuplicateRuleError(RegistryError):
    def __init__(self, groups: frozenset[str], existing_rule_name: str) -> None:
        a, b = sorted(groups)
        super().__init__(
            f"A reaction rule for ('{a}', '{b}') already exists: '{existing_rule_name}'."
        )


class BuildError(RuntimeError):
    pass


class NoReactiveSitesError(BuildError):
    def __init__(self, smiles: str) -> None:
        super().__init__(
            f"No reactive sites detected on monomer with SMILES '{smiles}'."
        )


class InfeasiblePolymerisationError(BuildError):
    pass


class AmbiguousBondingError(BuildError):
    def __init__(self, chain_group: str, monomer_group: str, candidate_count: int) -> None:
        super().__init__(
            f"{candidate_count} compatible pairings between chain group '{chain_group}' "
            f"and incoming group '{monomer_group}'. "
            "Specify site_in_group or site_out_group in AdditionStep."
        )


class ReactionFailedError(BuildError):
    def __init__(self, smirks: str, chain_group: str, monomer_group: str) -> None:
        super().__init__(
            f"SMIRKS produced no products for groups ('{chain_group}', '{monomer_group}'): "
            f"{smirks!r}"
        )


class IncompatibleSitesError(BuildError):
    def __init__(self, chain_group: str | None, monomer_group: str | None) -> None:
        super().__init__(
            f"No compatible site pairing between chain group {chain_group!r} "
            f"and incoming monomer group {monomer_group!r}."
        )


class IncompleteConstructError(BuildError):
    def __init__(self, open_site_count: int) -> None:
        super().__init__(
            f"Cannot convert to Polymer: {open_site_count} OPEN attachment site(s) remain. "
            "Call builder.cap() and/or builder.saturate_open_ends() first."
        )


class FFError(RuntimeError):
    pass


class MissingFragmentError(FFError):
    def __init__(self, uncovered_atom_indices: list[int]) -> None:
        super().__init__(
            f"No FragmentDefinition covers mol atom indices: {uncovered_atom_indices}."
        )


class AcpypeError(FFError):
    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(f"acpype exited with code {returncode}.\n{stderr}")


class ParameterisationConflictError(FFError):
    def __init__(self, atom_index: int, conflicting_fragment_names: list[str]) -> None:
        super().__init__(
            f"Mol atom {atom_index} matched by multiple fragments: "
            f"{conflicting_fragment_names}."
        )
