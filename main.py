import pprint
import random

from polymer_md.building.conversion import MonomerToResidueConverter
from polymer_md.building.data_models.map_labels import MapLabels
from polymer_md.building.growing_polymer import AdditionPolymer
from polymer_md.core.caps import BuiltinCap
from polymer_md.core.monomer import Monomer
from polymer_md.utils.rdkit_helper import RDKitHelper


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def step(msg: str) -> None:
    print(f"  → {msg}")


if __name__ == "__main__":
    random.seed(42)
    styrene_monomer = Monomer(smiles="C=Cc1ccccc1", label="S")
    polyeth_monomer = Monomer(smiles="CC=C", label="P")

    styrene = MonomerToResidueConverter.convert(monomer=styrene_monomer)
    polyeth = MonomerToResidueConverter.convert(monomer=polyeth_monomer)

    residue_list = [styrene, polyeth]
    polymer = AdditionPolymer()
    polymer.initialise(styrene, site=MapLabels.HEAD)

    n = 10
    for i in range(0, n):
        polymer.add(residue=residue_list[i % 2])

    polymer = polymer.export(cap=BuiltinCap.HYDROGEN)

    RDKitHelper.visualize_mol(mol=polymer.mol)
    pprint.pp(polymer)
