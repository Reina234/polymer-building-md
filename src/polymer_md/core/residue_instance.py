from dataclasses import dataclass


@dataclass(frozen=True)
class ResidueInstance:
    residue_id: str
    instance_number: int
    atom_indices: frozenset[int]
