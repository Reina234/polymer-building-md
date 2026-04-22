from __future__ import annotations

from dataclasses import dataclass

from polymer_md.parameterisation.data_models.parameterised_trimer import ParameterisedTrimer
from polymer_md.parameterisation.fragments.data_models.fragment import Fragment
from polymer_md.parameterisation.fragments.extraction.region_extractor import RegionFragmentExtractor
from polymer_md.utils.parmed_helper import CoordinateCrosswalk, StructureMolDeriver


@dataclass
class TerminalFragmentExtractor:
    def extract(self, parameterised_trimer: ParameterisedTrimer) -> list[Fragment]:
        derived_mol = StructureMolDeriver.derive(parameterised_trimer.structure)
        mol3d_to_parmed = CoordinateCrosswalk.map_mol3d_to_parmed(
            parameterised_trimer.mol_3d,
            parameterised_trimer.structure,
        )
        left_parmed_indices = RegionFragmentExtractor.resolve_parmed_indices(
            heavy_atom_indices=parameterised_trimer.trimer_result.left_atom_indices,
            mol_3d=parameterised_trimer.mol_3d,
            mol3d_to_parmed=mol3d_to_parmed,
        )
        right_parmed_indices = RegionFragmentExtractor.resolve_parmed_indices(
            heavy_atom_indices=parameterised_trimer.trimer_result.right_atom_indices,
            mol_3d=parameterised_trimer.mol_3d,
            mol3d_to_parmed=mol3d_to_parmed,
        )
        extractor = RegionFragmentExtractor()
        return extractor.extract(
            derived_mol=derived_mol,
            structure=parameterised_trimer.structure,
            region_parmed_indices=left_parmed_indices,
        ) + extractor.extract(
            derived_mol=derived_mol,
            structure=parameterised_trimer.structure,
            region_parmed_indices=right_parmed_indices,
        )
