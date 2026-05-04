# polymer-md

Force-field parameterisation of stochastic addition copolymers with GAFF2, producing
GROMACS-ready topology files from monomer SMILES strings.

---

## How it works

`polymer-md` builds a reusable **fragment library** from short trimers, then *tiles* those parameters onto any polymer assembled from the same monomers.

```text
  MonomerSpec list          weights + SMILES for each monomer type
         │
         ▼
  Transition matrix         Markov chain over reactive sites (head / tail)
         │                  controls composition and HT/TT regioregularity
         ▼
  Polymer sequence          random chain of n monomers built from the matrix
         │
         ▼
  Unique trimer contexts    every (left – central – right) environment in the chain
         │
         ▼
  GAFF2 via acpype          AM1-BCC charges + atom types for each trimer
         │
         ▼
  Fragment library          SMARTS-keyed parameters, reusable across sequences
         │
         ▼
  Tiling                    match library patterns onto the full polymer structure
         │
         ▼
  GROMACS output            .itp / .gro / .top  +  ParameterisedMolecule
```

The key insight is that GAFF2 atom types and AM1-BCC charges depend on local chemical
environment — roughly 2–3 bonds away. A trimer captures exactly that context for the
central monomer, so parameters extracted from trimers are valid for any chain position
with the same left–central–right neighbour combination.

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

---

## Methodology

### Monomers and reactive sites

Each vinyl monomer has two reactive sites — the two carbons of the double bond:

| Site | Name | Typical position |
|---|---|---|
| 0 | **Head** (H) | α-carbon — the substituted end (e.g. carries the ester group in methacrylates) |
| 1 | **Tail** (T) | β-carbon — the backbone CH₂ end |

Monomers are built from SMILES via `MonomerToResidueConverter`, which produces an `AdditionPolymerResidue` with atom-map numbers marking the two sites. A `MonomerSpec`
pairs a residue with a **composition weight**.

### Head-to-tail addition and regioregularity

When two monomers connect, the orientation is described by a `BondType(from_site, to_site)`:

```
HT  (head-to-tail)   site 0 → site 1   strongly preferred in radical polymerisation
TT  (tail-to-tail)   site 1 → site 1   regioirregular
HH  (head-to-head)   site 0 → site 0   regioirregular
```

`ht_fraction` in `ProportionalSolver` controls the fraction of HT additions:
`ht_fraction=0.95` gives a predominantly HT chain with 5% irregular insertions;
`ht_fraction=1.0` gives a perfectly regioregular chain.

### Transition matrix and chain growth

Chain growth is modelled as a **first-order Markov chain** over reactive sites.
For N monomers there are 2N sites (head and tail of each).

`ProportionalSolver` assigns each site a probability proportional to its monomer's composition weight, scaled by `ht_fraction` for tail sites and `(1 − ht_fraction)` for
head sites. All rows of the matrix are identical, so the next addition depends only on composition and regioregularity — not on history.

`RandomPolymerBuilder` grows the chain one monomer at a time:

1. Sample the first site from the matrix's stationary distribution.
2. From the current active end, sample the next incoming site.
3. Connect the monomers, flip to the complementary site of the new monomer (the new active end).
4. Repeat until the chain has n monomers, then cap both ends with methyl groups (–CH₃).

**Methyl caps** are used rather than bare H termini because they provide a realistic sp3 chemical environment — consistent with how the chain interior looks to antechamber — and are the standard approach in GAFF polymer workflows.

### Trimer-based parameterisation

GAFF2 atom typing and AM1-BCC charge assignment both depend on the local environment up to roughly 2–3 bonds away. A **trimer** — one left, one central, and one right monomer — captures exactly this context:

```
 [cap]──── left ────[bond]──── central ────[bond]──── right ────[cap]
                                  ↑
                       parameters extracted here
```

Parameters from the central monomer region are valid for any chain position that has the same left–central–right neighbour combination and the same bond orientations (HT/TT). Trimers are therefore a compact, reusable representation of the full parameter space.

Each unique trimer is processed once:

```
TrimerResult mol
  → ETKDG conformer  (ETKDGConformerGenerator)
  → SDF → MOL2       (OBabelConverter)
  → MOL2 → ITP/GRO   (AcpypeConverter, antechamber, AM1-BCC charges)
  → parmed structure
```

Results are **cached on disk** by a content-addressed name
`{left}_{central}_{right}_{bonds}_{smiles-hash}`. Re-running the pipeline loads the cached files without invoking antechamber again. A shared cache across runs can be set with `trimer_cache_dir`.

An in-memory SMILES deduplication step within each run ensures that chemically identical trimers with different monomer labels are only parameterised once.

### Oriented triplets and the polymer-first approach

A copolymer with three monomers and irregular additions could in principle require up to 3 × 3 × 3 × (2³) = 216 distinct oriented triplets
`(left_id, left_bond_site, central_id, incoming_site, right_id, right_bond_site)`, but any given 20-mer sequence uses only a small subset.

**`PolymerFirstParameterisationPipeline`** exploits this:

1. Build the target polymer sequence first.
2. Walk the chain and collect the unique oriented triplets that actually appear.
3. Parameterise only those trimers — typically 5–15 for a 20-mer with three monomers.

**`PolymerParameterisationPipeline`** is the alternative: it enumerates triplets from the
transition matrix without first building the polymer, useful when the library will be reused
across many different chains from the same monomer set.

### Fragment library extraction

From each parameterised trimer, `RegionFragmentExtractor` builds portable parameter records:

1. **SMARTS subgraph**: `SmartsBuilder` runs a depth-first traversal from each atom in the
   target region, encoding element, aromaticity, ring membership, and degree.
2. **Annotated members**: `AnnotatedAtom`, `AnnotatedBond`, `AnnotatedAngle`, and
   `AnnotatedDihedral` objects map local SMARTS indices to specific force-field terms.
3. **Values**: `FragmentMatcher` reads the corresponding values from the parmed structure
   (charges, LJ ε/σ, bond k and req, angle k and θeq, dihedral terms).

The `FragmentLibrary` maps `SMARTS pattern → local indices → parameter values` and is
JSON-serialisable for persistence and reuse.

### Tiling

`PolymerParameterisationTiler` applies the library to the target polymer:

| Step | Detail |
|---|---|
| Match patterns | `GetSubstructMatches` for every library SMARTS against the polymer mol |
| Collect values | For each `(global_indices, parameter)` pair, gather all matched values |
| Resolve conflicts | Multiple matches → `MeanStrategy` (arithmetic mean) by default |
| Handle misses | No match → invoke the registered `MissingParameterStrategy` for that type |
| Hydrogen types | Infer GAFF2 types from the bonded heavy atom (`hc`, `ha`, `ho`, `hn`, …) after all heavy atoms are typed |
| Charge neutrality | Distribute any residual charge uniformly across all atoms |

### Missing parameter strategies

When no library pattern covers a particular bond, angle, or atom:

| Strategy | Behaviour |
|---|---|
| `StrictMissingParameterStrategy` | Raise `MissingParameterError` (default — ensures full coverage) |
| `ResiduePositionStrategy` | Look up by (residue_id, position-within-residue) |
| `MinimalMoleculeStrategy` | Extract a minimal chemical fragment, run antechamber on it |
| `NeighbourhoodSMARTSStrategy` | Search by progressively broader SMARTS environments |

Strategies are registered **per parameter type**, so atoms can use a different fallback than
bonds or angles:

```python
missing_strategies={AtomParameter: ResiduePositionStrategy(min_matches=1)}
```

---

## Architecture

```
src/polymer_md/
│
├── core/                    Pure data; RDKit only
│   ├── monomer.py           Monomer (SMILES + label)
│   ├── polymer.py           Polymer (mol + ResidueInstance list)
│   ├── residue_instance.py  ResidueInstance (id, type, atom tag set)
│   └── caps.py              BuiltinCap.METHYL, Cap protocol
│
├── building/                Chain construction
│   ├── monomer_converter.py MonomerToResidueConverter
│   ├── random_polymer.py    RandomPolymerBuilder  →  Polymer + connection list
│   ├── trimer_builder.py    TrimerBuilder  →  TrimerResult list
│   ├── growing_polymer.py   AdditionPolymer  (internal step-growth assembly)
│   ├── solvers/
│   │   ├── base.py          TransitionMatrixSolver protocol
│   │   └── proportional.py  ProportionalSolver  (ht_fraction)
│   └── data_models/
│       ├── monomer_spec.py  MonomerSpec  (residue + weight)
│       ├── transition_matrix.py  TransitionMatrix, SiteKey
│       └── trimer.py        TrimerResult, BondType  (HT / TT / HH labels)
│
├── geometry/                Conformer generation  (injected into pipelines)
│   └── etkdg.py             ETKDGConformerGenerator  (use_uff, random_seed)
│
├── conversion/              Subprocess wrappers; no chemistry logic
│   ├── obabel.py            OBabelConverter  (SDF → MOL2)
│   ├── acpype.py            AcpypeConverter  (MOL2 → ITP/GRO/TOP)
│   └── gromacs_files.py     GromacsFiles  (itp, gro, top paths)
│
├── parameterisation/
│   ├── fragments/
│   │   ├── extraction/
│   │   │   ├── region_extractor.py  Core: context mol → SMARTS records
│   │   │   ├── interior.py          InteriorFragmentExtractor  (central region)
│   │   │   ├── terminal.py          TerminalFragmentExtractor  (left/right + cap)
│   │   │   ├── smarts_builder.py    DFS SMARTS subgraph builder
│   │   │   └── builder.py           FragmentLibraryBuilder
│   │   ├── matching/
│   │   │   ├── matcher.py           FragmentMatcher  (SMARTS → parmed values)
│   │   │   └── resolution.py        MeanStrategy
│   │   ├── library.py               FragmentLibrary  (SMARTS store, JSON I/O)
│   │   └── data_models/             Fragment, AnnotatedMember types, ParameterRecord
│   ├── strategies/
│   │   ├── strict.py                StrictMissingParameterStrategy
│   │   ├── residue_position.py      ResiduePositionStrategy
│   │   ├── minimal_molecule.py      MinimalMoleculeStrategy
│   │   └── neighbourhood_smarts.py  NeighbourhoodSMARTSStrategy
│   ├── tiler.py                     PolymerParameterisationTiler
│   ├── pipeline.py                  TrimerParameterisationPipeline  (disk cache)
│   ├── polymer_pipeline.py          Orchestrator A: library → polymer
│   └── polymer_first_pipeline.py    Orchestrator B: polymer-first → minimal trimers
│
├── analysis/
│   ├── comparator.py        ParameterComparator  (multi-molecule SMARTS comparison)
│   ├── extraction_spec.py   ExtractionSpec  (pattern + parameters → Fragment)
│   ├── atom_deviation.py    AtomDeviation, FragmentDeviationSummary
│   └── report.py            TextReport
│
├── visualisation/
│   ├── polymer_3d.py        PolymerViewer  (interactive 3D: charge / type / residue colouring)
│   ├── difference_3d.py     DifferenceViewer  (atom view + CG bead view, deviation colouring)
│   ├── trimer.py            plot_parameterised_trimer  (2D structure + charge bar)
│   ├── polymer.py           plot_parameterised_polymer  (2D structure + histogram)
│   └── comparison.py        plot_comparison  (box plots across molecules)
│
└── utils/
    ├── parmed_helper.py     ParmedTypeResolver, mol_from_structure
    ├── topology_builder.py  TopologyBuilder  (RDKit mol → bare parmed Structure)
    └── rdkit_helper.py      RDKitHelper  (join mols, canonical SMILES, SMARTS)
```

**Dependency order (strictly acyclic):**

```
core ← geometry ← building ← conversion ← parameterisation ← analysis ← visualisation
```

---

## Key design decisions

**SMARTS as the parameter index.** Parameters are stored under the chemical pattern that produced them, not under an atom position or residue name. This makes the library transferable: a pattern that matched a C–C bond in an MMA–BA–MMA trimer will match the same environment anywhere in any polymer built from those monomers.

**Index identity between RDKit and parmed.** `TopologyBuilder.build(mol_3d)` adds parmed atoms in the same order as RDKit iterates them, so `structure.atoms[i].idx == i` and the polymer mol maps directly to the parmed structure without coordinate cross-walking.

**Cap atoms are in the library.** `TerminalFragmentExtractor` extracts cap atoms as their
own region, so their charges and LJ parameters come from the library rather than being
silently defaulted.

**Hydrogen types are inferred post-tiling.** Fragment SMARTS patterns cover heavy atoms only. After all heavy atoms are typed, hydrogen GAFF2 types are assigned from the bonded heavy atom's type — `hc` for H on sp3 carbon, `ha` for H on aromatic carbon, `ho` for H on oxygen, and so on.

---

## Known limitations

| Issue | Severity |
|---|---|
| Only the first Fourier dihedral term is stored; GAFF2 commonly assigns 2–4 terms | Critical |
| Improper dihedrals not extracted — sp2 planarity (carbonyls, aromatic rings) not enforced | High |
| Minimal-molecule expansion is not ring-aware — aromatic monomers may receive wrong atom types | High |
| Cross-run trimer caching requires explicitly setting `trimer_cache_dir` | Medium |

---

## Tests

```bash
uv run pytest                          # ~470 tests, ~15 s
uv run pytest --cov=src/polymer_md     # with coverage
uv run pytest tests/parameterisation/  # parameterisation subsystem only
```
