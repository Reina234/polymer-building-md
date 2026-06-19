"""
Parameterise every polymer composition using PolymerParameterisationPipeline
and write full-chain GROMACS files (.gro, .top) for each.

Compositions are hardcoded below as wt% dicts (from compositions_polymer.xlsx)
and converted to mole fractions before being passed to MonomerSpec.

Checkpointing
-------------
A JSON file at output/checkpoint.json tracks which polymer IDs have completed
successfully. On re-run, completed polymers are skipped automatically.
Delete output/checkpoint.json (or individual IDs within it) to re-run.

Run:
    PATH="/Users/reinazheng/miniconda3/envs/md_engines/bin:$PATH" uv run python run_all_polymers.py
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.building.solvers.proportional import ProportionalSolver
from polymer_md.core.monomer import Monomer
from polymer_md.geometry.etkdg import ETKDGConformerGenerator
from polymer_md.parameterisation.fragments.data_models.parameters import AtomParameter
from polymer_md.parameterisation.polymer_pipeline import PolymerParameterisationPipeline
from polymer_md.parameterisation.strategies.residue_position import (
    ResiduePositionStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N = 25  # repeat units per chain
TRIMER_CACHE_DIR = Path("output/_trimer_cache")  # shared across all polymers

# ---------------------------------------------------------------------------
# Monomer SMILES + repeat-unit MW (g/mol)
# ---------------------------------------------------------------------------

MONOMERS: dict[str, tuple[str, float]] = {
    "STY": ("C=Cc1ccccc1", 104.15),
    "EA": ("C=CC(=O)OCC", 100.12),
    "BA": ("C=CC(=O)OCCCC", 128.17),
    "MMA": ("C=C(C)C(=O)OC", 100.12),
    "MAA": ("C=C(C)C(=O)O", 86.09),
    "AA": ("C=CC(=O)O", 72.06),
    "BMA": ("C=C(C)C(=O)OCCCC", 142.20),
    "2-EHA": ("C=CC(=O)OCC(CC)CCCC", 184.28),
    "iBMA": ("C=C(C)C(=O)OCC(C)C", 142.20),
    "EMA": ("C=C(C)C(=O)OCC", 114.14),
}

# ---------------------------------------------------------------------------
# Polymer compositions (wt%) from compositions_polymer.xlsx
# No.  STY   EA    BA    MMA   MAA   AA    BMA   2-EHA  iBMA   EMA   Tg
# ---------------------------------------------------------------------------

POLYMERS: list[dict] = [
    {"id": 1,  "tg": 49.80, "wt": {"EA": 44.5, "MMA": 54.5, "MAA": 1.0}},
    {"id": 2,  "tg": 65.48, "wt": {"MMA": 37.2, "MAA": 0.9, "BMA": 61.9}},
    {"id": 3,  "tg": 62.38, "wt": {"MMA": 34.6, "MAA": 1.0, "BMA": 64.4}},
    {"id": 4,  "tg": 56.23, "wt": {"MMA": 20.0, "BMA": 80.0}},
    {"id": 5,  "tg": 69.61, "wt": {"MMA": 80.0, "BMA": 20.0}},
    {"id": 6,  "tg": 57.10, "wt": {"EA": 29.6, "MMA": 49.2, "MAA": 1.5, "EMA": 19.7}},
    {"id": 7,  "tg": 64.15, "wt": {"MMA": 37.1, "MAA": 1.0, "BMA": 61.9}},
    {"id": 8,  "tg": 57.09, "wt": {"2-EHA": 10.0, "iBMA": 90.0}},
    {"id": 9,  "tg": 62.06, "wt": {"STY": 50.5, "AA": 1.0, "2-EHA": 12.9, "iBMA": 35.6}},
    {"id": 10, "tg": 81.50, "wt": {"MMA": 66.0, "AA": 0.5, "BMA": 33.5}},
    {"id": 11, "tg": 68.50, "wt": {"BA": 19.0, "MMA": 78.0, "AA": 3.0}},
    {"id": 12, "tg": 72.40, "wt": {"BA": 19.0, "MMA": 78.0, "MAA": 3.0}},
    {"id": 13, "tg": 66.70, "wt": {"STY": 77.0, "AA": 3.0, "2-EHA": 20.0}},
    {"id": 14, "tg": 66.00, "wt": {"STY": 77.0, "MAA": 3.0, "2-EHA": 20.0}},
]

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path("output/checkpoint.json")


def load_checkpoint() -> set[int]:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data["completed"])
    return set()


def save_checkpoint(completed: set[int]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_specs(wt_dict: dict[str, float]) -> list[MonomerSpec]:
    """Convert wt% composition → mole-fraction MonomerSpecs.

    n_i = wt_i / MW_i,  x_i = n_i / sum(n_j)
    """
    molar = {label: wt / MONOMERS[label][1] for label, wt in wt_dict.items()}
    total = sum(molar.values())
    return [
        MonomerSpec(
            residue=MonomerToResidueConverter.convert(
                Monomer(smiles=MONOMERS[label][0], label=label)
            ),
            weight=n / total,
        )
        for label, n in molar.items()
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    completed = load_checkpoint()

    if completed:
        print(f"Resuming — already completed: {sorted(completed)}")

    failed: list[int] = []

    for polymer in POLYMERS:
        polymer_id = polymer["id"]
        tg = polymer["tg"]

        print(f"\n{'='*60}")

        if polymer_id in completed:
            print(f"Polymer {polymer_id:>3}  [SKIP — already completed]")
            continue

        print(f"Polymer {polymer_id:>3}  (Tg = {tg} °C)")

        specs = to_specs(polymer["wt"])
        for s in specs:
            print(f"  {s.residue.label:<6}  {s.weight:.4f} mol-frac")

        output_dir = Path("output") / f"polymer_{polymer_id:03d}"
        print(f"  output → {output_dir}")

        try:
            pipeline = PolymerParameterisationPipeline(
                specs=specs,
                n=N,
                output_dir=output_dir,
                seed=42,
                solver=ProportionalSolver(ht_fraction=0.95),
                conformer_generator=ETKDGConformerGenerator(
                    use_uff=False, use_random_coords=True
                ),
                missing_strategies={
                    AtomParameter: ResiduePositionStrategy(min_matches=1)
                },
                adjust_charge=True,
                charge_method="bcc",
                trimer_cache_dir=TRIMER_CACHE_DIR,
            )
            result = pipeline.run()

            total_charge = sum(a.charge for a in result.structure.atoms)
            print(f"  Atoms:        {len(result.structure.atoms)}")
            print(f"  Bonds:        {len(result.structure.bonds)}")
            print(f"  Total charge: {total_charge:.4f} e")
            print(f"  GRO:          {result.source.gro}")
            print(f"  TOP:          {result.source.top}")

            completed.add(polymer_id)
            save_checkpoint(completed)

        except Exception:
            print(f"  [ERROR] Polymer {polymer_id} failed — will retry on next run")
            traceback.print_exc()
            failed.append(polymer_id)

    print(f"\n{'='*60}")
    if failed:
        print(f"Done. Failed (will retry on re-run): {failed}")
    else:
        print("Done. All polymers completed successfully.")


if __name__ == "__main__":
    main()
