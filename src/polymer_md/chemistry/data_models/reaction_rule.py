from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ReactionOutcome:
    reaction_smarts: str
    weight: float = 1.0


@dataclass(frozen=True)
class ReactionRule:
    name: str
    groups: frozenset[str]
    outcomes: tuple[ReactionOutcome, ...]

    def sample(self, rng: random.Random) -> ReactionOutcome:
        options = list(self.outcomes)
        weights = [outcome.weight for outcome in options]
        return rng.choices(options, weights=weights, k=1)[0]
