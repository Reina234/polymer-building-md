from polymer_md.chemistry.data_models import GroupType, ReactionOutcome, ReactionRule
from polymer_md.chemistry.registry import ChemistryRegistry

CARBOXYLIC_ACID = GroupType(
    name="carboxylic_acid",
    smarts="[CX3](=O)[OX2H1]",
    overrides=frozenset({"hydroxyl"}),
)

HYDROXYL = GroupType(
    name="hydroxyl",
    smarts="[OX2H1]",
)

PRIMARY_AMINE = GroupType(
    name="primary_amine",
    smarts="[NX3H2]",
)

VINYL = GroupType(
    name="vinyl",
    smarts="[CX2H2]=[CX2H1,CX2H0]",
)

ESTER_BOND = ReactionRule(
    name="ester_bond",
    groups=frozenset({"carboxylic_acid", "hydroxyl"}),
    outcomes=(
        ReactionOutcome(
            reaction_smarts="[C:1](=[O:2])[OX2H1:3].[OX2H1:4]>>[C:1](=[O:2])[O:4]",
        ),
    ),
)

AMIDE_BOND = ReactionRule(
    name="amide_bond",
    groups=frozenset({"carboxylic_acid", "primary_amine"}),
    outcomes=(
        ReactionOutcome(
            reaction_smarts="[C:1](=[O:2])[OX2H1:3].[NX3H2:4]>>[C:1](=[O:2])[N:4]",
        ),
    ),
)


def build_builtin_registry() -> ChemistryRegistry:
    registry = ChemistryRegistry()
    for group in [CARBOXYLIC_ACID, HYDROXYL, PRIMARY_AMINE, VINYL]:
        registry.add_group(group)
    for rule in [ESTER_BOND, AMIDE_BOND]:
        registry.add_rule(rule)
    return registry


BUILTIN_REGISTRY = build_builtin_registry()
