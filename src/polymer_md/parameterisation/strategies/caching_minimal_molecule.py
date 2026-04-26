from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from polymer_md.parameterisation.fragments.data_models.parameters import DihedralTerm, ForceFieldParameter
from polymer_md.parameterisation.strategies.base import MissingParameterStrategy, StrategyContext
from polymer_md.parameterisation.strategies.minimal_molecule import MinimalMoleculeStrategy


@dataclass
class CachingMinimalMoleculeStrategy:
    inner: MinimalMoleculeStrategy = field(default_factory=MinimalMoleculeStrategy)
    _cache: dict[str, float | tuple[DihedralTerm, ...]] = field(default_factory=dict, init=False, repr=False)

    def resolve(
        self,
        global_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
        context: StrategyContext,
    ) -> float | tuple[DihedralTerm, ...]:
        expanded = self.inner.expander.expand(context.derived_mol, frozenset(global_indices))
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(context.derived_mol, expanded)
        local_indices = tuple(old_to_new[i] for i in global_indices)
        cache_key = self._build_key(mol, local_indices, parameter)
        if cache_key in self._cache:
            return self._cache[cache_key]
        value = self.inner.resolve(global_indices, parameter, context)
        self._cache[cache_key] = value
        return value

    @staticmethod
    def _build_key(
        mol: Chem.Mol,
        local_indices: tuple[int, ...],
        parameter: ForceFieldParameter,
    ) -> str:
        canonical = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
        return f"{canonical}|{local_indices}|{type(parameter).__name__}.{parameter.name}"
