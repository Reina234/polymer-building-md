
This repo builds a reusable fragment library from short trimers, then tiles those parameters onto any polymer assembled from the same monomers.

## Why?

Parameterising a whole polymer chain with antechamber is slow, and you have to redo it every time the length or composition changes. So this doesn't bother with the whole chain. It parameterises a few short trimers, saves their parameters as a reusable fragment library, and then tiles them onto a polymer of any length built from the same monomers. You get GROMACS-ready `.itp` / `.gro` / `.top` files at the end. This is a small refactored module pulled out of my IIB project, which chains this with reusing explicit solvent environments, and course graining modules.

This currently works for homopolymers and random copolymers, with more to come. There are also tools for checking the result against a reference topology, including a 3D viewer that shows where the
parameters differ.


---

## Installation

Both `antechamber` (AmberTools) and `obabel` (Open Babel) are available through conda:

```bash
conda create -n md_engines -c conda-forge ambertools openbabel
conda activate md_engines
```

The Python environment is managed separately with `uv`:

```bash
uv sync
source .venv/bin/activate
```

Scripts that invoke antechamber or obabel must be run with the conda environment on the PATH:

```bash
PATH="$(conda info --base)/envs/md_engines/bin:$PATH" uv run python examples/your_script.py
```

---

## Quick start
-> run_all_polymers.py

```python
from pathlib import Path
from polymer_md.building.data_models.monomer_spec import MonomerSpec
from polymer_md.building.monomer_converter import MonomerToResidueConverter
from polymer_md.core.monomer import Monomer
from polymer_md.parameterisation.polymer_first_pipeline import PolymerFirstParameterisationPipeline

specs = [
    MonomerSpec(
        residue=MonomerToResidueConverter.convert(Monomer(smiles="C=C(C)C(=O)OC", label="MMA")),
        weight=0.7,
    ),
    MonomerSpec(
        residue=MonomerToResidueConverter.convert(Monomer(smiles="C=CC(=O)OCCCC", label="BA")),
        weight=0.3,
    ),
]

molecule, library = PolymerFirstParameterisationPipeline(
    specs=specs,
    n=20,
    output_dir=Path("output/my_polymer"),
    seed=42,
    charge_method="bcc",
).run()
# output/my_polymer/polymer_20mer.itp / .gro / .top
```

```python
# 3D visualisation
from polymer_md.visualisation.polymer_3d import PolymerViewer
PolymerViewer(molecule).save(Path("output/polymer.html"))
```

```python
# Comparison against a reference topology
from polymer_md.analysis.extraction_spec import ExtractionSpec
from polymer_md.parameterisation.fragments.data_models.parameters import BondParameter, AtomParameter
from polymer_md.visualisation.difference_3d import DifferenceViewer

fragments = [
    ExtractionSpec("[#6;A]-[#6;A]", (BondParameter.FORCE_CONSTANT,)).to_fragment(),
    ExtractionSpec("[#6;A]",        (AtomParameter.CHARGE,)).to_fragment(),
]
DifferenceViewer(molecule, reference, fragments).save(Path("output/diff.html"))
```
