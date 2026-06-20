
`polymer-md` builds a reusable **fragment library** from short trimers, then *tiles* those parameters onto any polymer assembled from the same monomers.


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
