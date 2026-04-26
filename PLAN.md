# polymer-md — Developer Reference and Improvement Plan

---

## 1. Repository Purpose

A Python library for force-field parameterisation of stochastic copolymers.

The core idea: rather than parameterising the full polymer at once (expensive and inflexible),
fragment short trimers (left–central–right monomer) with GAFF2 via acpype, extract parameters
as portable SMARTS-keyed records, and tile them onto any polymer built from the same monomer set.
The output is a GROMACS-ready `.itp`/`.gro`/`.top` and a `ParameterisedMolecule` for downstream
analysis.

---

## 2. Module Map

```
src/polymer_md/
│
├── core/
│   Monomer, Polymer, ResidueInstance, Cap, residue tagging.
│   Pure data — no external deps beyond rdkit. Foundation of everything.
│
├── geometry/
│   ConformerGenerator protocol + ETKDGConformerGenerator (OBabel variant also available).
│   Injected into pipelines; never imported directly by other modules.
│
├── building/
│   TransitionMatrixSolver → RandomPolymerBuilder → Polymer.
│   TrimerBuilder → all reachable trimers above p_min.
│   TrimerResult: holds left/central/right/cap atom index sets (mol-space).
│
├── conversion/
│   OBabelConverter (SDF → MOL2), AcpypeConverter (MOL2 → GROMACS), GromacsFiles.
│   Thin subprocess wrappers. No chemistry logic here.
│
├── parameterisation/
│   │
│   ├── fragments/
│   │   ├── data_models/
│   │   │   Fragment (SMARTS + AnnotatedMembers), ParameterRecord,
│   │   │   ParameterHit, FragmentMatch, AtomMetadata.
│   │   │
│   │   ├── extraction/
│   │   │   RegionFragmentExtractor   — core: given context + region parmed indices,
│   │   │                              builds SMARTS subgraph and extracts annotated members.
│   │   │   InteriorFragmentExtractor — wraps RegionFragmentExtractor for central-monomer role.
│   │   │   TerminalFragmentExtractor — wraps it for left/right-monomer + cap roles.
│   │   │                              (cap atoms now extracted as their own region.)
│   │   │   SmartsBuilder             — DFS-based canonical SMARTS subgraph builder.
│   │   │   FragmentLibraryBuilder    — assembles all fragment pairs into a FragmentLibrary.
│   │   │
│   │   ├── matching/
│   │   │   FragmentMatcher      — SMARTS substructure matching + parameter value extraction.
│   │   │   resolution.py        — ResolutionStrategy (MeanStrategy default).
│   │   │
│   │   └── library.py
│   │       FragmentLibrary      — SMARTS-keyed store, JSON-serialisable, query interface.
│   │
│   ├── strategies/
│   │   MissingParameterStrategy protocol + implementations:
│   │   StrictMissing, ResiduePosition, NeighbourhoodSMARTS, MinimalMolecule.
│   │   Injected per parameter-type into the tiler.
│   │
│   ├── data_models/
│   │   ParameterisedTrimer (trimer_result + structure + mol_3d + gromacs_files).
│   │   ParameterisedMolecule (structure + mol + source).
│   │
│   ├── pipeline.py                TrimerParameterisationPipeline
│   ├── fragment_library_pipeline.py  Orchestrator 1: monomers → FragmentLibrary file
│   ├── polymer_pipeline.py           Orchestrator 2: library + specs → ParameterisedMolecule
│   └── tiler.py
│       PolymerParameterisationTiler — SMARTS-matches library onto polymer structure;
│       dispatches to MissingParameterStrategy per type when no match found.
│
├── analysis/
│   ParameterComparator, ExtractionSpec, ComparisonResult, Summary.
│   Reads only from ParameterisedMolecule; no dep on building or extraction.
│
├── visualisation/
│   plot_parameterised_trimer  — 2D structure (colored by region) + charge bar chart.
│   plot_parameterised_polymer — 2D structure (colored by charge) + histogram + stats.
│   plot_comparison            — box plots across labeled molecules per (fragment, parameter).
│   view_trimer_3d / view_polymer_3d — py3Dmol; auto-detects Jupyter vs browser.
│
└── utils/
    ParmedTypeResolver     — safe gateway to bond.type / angle.type / dihedral.type.
    CoordinateCrosswalk    — nearest-neighbour mol_3d → parmed index mapping.
    StructureMolDeriver    — parmed Structure → RDKit Mol (PDB round-trip + DetermineBonds).
    TopologyBuilder        — RDKit Mol → bare parmed Structure (atoms + bonds + angles + dihedrals).
```

**Dependency order (acyclic, enforced):**
```
core ← geometry ← building ← conversion ← parameterisation ← analysis ← visualisation
```

---

## 3. Data Flow

### Pipeline 1 — Fragment Library Building

```
MonomerSpec list
  → TransitionMatrixSolver → TrimerBuilder → filter by p_min
  → for each TrimerResult:
      ETKDGConformerGenerator.embed(trimer.mol)  → mol_3d
      OBabelConverter: mol_3d SDF → MOL2
      AcpypeConverter: MOL2 → GROMACS (ITP/GRO/TOP)
      ParameterisedTrimer(trimer_result, structure, mol_3d, gromacs_files)
  → InteriorFragmentExtractor  (region = central monomer)
  → TerminalFragmentExtractor  (region = left / right monomer + cap atoms)
  → FragmentLibraryBuilder     → FragmentLibrary
  → FragmentLibrary.save(path)
```

### Pipeline 2 — Polymer Parameterisation

```
FragmentLibrary + MonomerSpecs + n
  → RandomPolymerBuilder → Polymer
  → ETKDGConformerGenerator.embed(polymer.mol) → mol_3d
  → TopologyBuilder.build(mol_3d) → pmd.Structure  (topology only, no params)
  → StructureMolDeriver.derive(structure) → derived_mol  (for SMARTS matching)
  → _build_polymer_atom_metadata(polymer)  (monomer heavy-atom positions, excl. caps)
  → PolymerParameterisationTiler.tile(structure, derived_mol, metadata)
      FragmentMatcher.match_all(derived_mol) → hits per (global_indices, parameter)
      per atom/bond/angle/dihedral:
        if library hit → ResolutionStrategy.resolve(values)
        else if atom in metadata → MissingParameterStrategy[AtomParameter].resolve(...)
        else → log WARNING, keep parmed default
  → adjust_charge_neutrality(structure)
  → _save_gromacs(structure) → GromacsFiles
  → ParameterisedMolecule(structure, mol_3d, gromacs_files)
```

### Pipeline 3 — Analysis

```
ExtractionSpec(pattern, parameters) → .to_fragment() → Fragment
ParameterComparator(fragments)
  → .compare({"label": ParameterisedMolecule, ...}) → ComparisonResult
  → .summarise() → {label → fragment → parameter → Summary(mean, std, count, min, max)}
```

---

## 4. Key Design Rules

| Rule | Detail |
|---|---|
| Constructor injection everywhere | No globals. Pipelines receive conformer, charge method, strategies, solver via `__init__`. |
| `ParmedTypeResolver` is the only gateway | Never access `bond.type.k` directly. Always use `ParmedTypeResolver.bond_type(bond).k`. Handles `None` and `DihedralTypeList`. |
| `derived_mol` is the canonical polymer mol | Index `i` in `derived_mol` == `structure.atoms[i]` (preserved by PDB write order). All SMARTS matching uses `derived_mol`. |
| `CoordinateCrosswalk` bridges mol spaces | Maps pre-acpype mol_3d indices → post-acpype parmed indices by nearest-coordinate. Required any time you compare `TrimerResult` region sets against `structure`. |
| Strategies are per parameter type | `missing_strategies: dict[type[ForceFieldParameter], MissingParameterStrategy]`. Atom, Bond, Angle, Dihedral can each have different fallbacks. |
| `polymer_atom_metadata` covers monomers only | Cap atoms and H atoms are deliberately excluded. The tiler safety net skips (rather than crashes) atoms with no library hit and no metadata. |

---

## 5. Academic / Chemical Validity

### What is validated and sound

**Trimer-based parameterisation** is established methodology. Short oligomers (trimers, 6-mers
with head/tail groups) parameterised via antechamber/AM1-BCC is the approach used by
PolymerGo, LigParGen-based polymer workflows, and published GAFF polymer studies.
Left/central/right context captures the chemical environment that determines GAFF2 atom types
and AM1-BCC charge distribution.

**AM1-BCC charges per trimer** are acceptable. Quality is comparable to HF/6-31G* ESP charges.
Cross-chain polarisation is modest for saturated polymers and is the standard approximation.
Conjugated/aromatic polymers may show larger effects but the trimer context still captures the
dominant nearest-neighbour electronic influence.

**Methyl caps** are the standard termination for saturated polymer chain ends. They provide a
realistic sp3 environment rather than the artificially electron-withdrawing H cap.

**Charge neutrality correction** (post-tiling) is correct. Per-trimer AM1-BCC charges do not
sum to zero for a hetero-chain; distributing the remainder per-atom is the accepted approach.

### Known limitations

| Issue | Severity | Impact |
|---|---|---|
| Dihedral multi-periodicity: only first Fourier term stored | **Critical** | Backbone rotation barriers can be wrong by 1–3 kcal/mol for bonds adjacent to sp2 atoms. GAFF2 routinely uses 2–4 terms. |
| Improper dihedrals skipped entirely | **High** | sp2 planarity not enforced. Aromatic rings, carbonyls, and planar N can pucker in long NPT runs. |
| Minimal molecule ring incompleteness | **High** | `MinimalMoleculeStrategy` adds only immediate neighbors. Aromatic atoms need the full ring for antechamber to assign `ca`/`cc`/`cd` correctly. Truncated ring → wrong atom type → wrong parameters. |
| 1-4 scaling not written to ITP | **Medium** | GAFF2 uses `scee=1.2`, `scnb=2.0`. These must appear in `[defaults]` or `[ pairs ]`. Currently absent. |
| No trimer or minimal-mol deduplication | **Low–Medium** | Re-running the library pipeline on overlapping monomer sets re-invokes acpype on identical trimers. Slow and wasteful. |
| Library schema has no version field | **Low** | Silent misread if schema changes (e.g. dihedral multi-periodicity addition). |

---

## 6. Improvement Plan

### 6.1 Dihedral multi-periodicity  *(schema-breaking, do first)*

GAFF2 assigns multiple Fourier terms to many dihedrals. A single-term approximation is a
quantitatively wrong representation of the torsion potential.

**Changes required:**

`FragmentLibrary` / `ParameterRecord`:
- Currently stores `float` values per `(global_indices, DihedralParameter)` triple.
- Change dihedral storage to `list[tuple[float, float, float]]` (phi_k, phase, per) per record.
- This affects serialisation: bump `schema_version` to `"2"` and add a migration reader.

`RegionFragmentExtractor._extract_dihedral_members`:
- Extract all Fourier terms per dihedral, not just the first.
- `FragmentMatcher._extract_value` for dihedrals returns all terms.

`PolymerParameterisationTiler._apply_dihedrals`:
- Assign a full `DihedralTypeList` to the parmed structure instead of a single `DihedralType`.

`ResolutionStrategy` for dihedrals:
- For multiple fragment matches, use the first-match term set (averaging Fourier series term-by-term is not physically meaningful).
- Add a `DihedralResolutionStrategy` distinct from scalar `ResolutionStrategy`.

`ParmedTypeResolver.dihedral_type`:
- Return `list[pmd.DihedralType]` (all terms), not just `dtype[0]`.

---

### 6.2 Improper dihedrals

GAFF2 uses impropers to enforce planarity of sp2 centres (aromatic rings, carbonyls, planar N).
Without them rings pucker in long NPT simulations.

**Changes required:**

`RegionFragmentExtractor`:
- Add `_extract_improper_members`. Improper convention in GAFF: the central sp2 atom is at
  position j; its three substituents are i, k, l (any order).
- Guard: only extract impropers where the central atom is sp2 (check
  `atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2` on `derived_mol`).

`Fragment` / `AnnotatedMember`:
- Add `AnnotatedImproper` alongside `AnnotatedDihedral`, or add an `improper: bool` flag.

`PolymerParameterisationTiler`:
- Add `_apply_impropers` alongside `_apply_dihedrals`.

`parmed` representation:
- Parmed uses `pmd.Dihedral(..., improper=True)` for impropers. The tiler currently
  skips these; it should assign them.

---

### 6.3 Minimal molecule strategy — ring-aware expansion  *(GAFF-specific)*

**GAFF2 atom type assignment rules relevant to minimal context:**

| Atom class | Needs | Example |
|---|---|---|
| sp3 C, H (ct, hc, …) | element + immediate neighbors | CH3–CH3 (ethane) sufficient |
| sp2 non-ring C (c, c2) | element + immediate neighbors | acetaldehyde sufficient |
| Aromatic C, N (ca, cc, na, …) | **full aromatic ring** | benzene; pyridine |
| Fused-ring atoms | **both rings** | naphthalene; indole |

The current `_extract_minimal_mol` adds only immediate neighbors for all atoms. For aromatic
polymers (polystyrene, polythiophene, polyaniline) this produces truncated rings, antechamber
assigns wrong types, and the retrieved parameters are wrong.

**Fix — ring-aware expansion:**

```python
def _extract_minimal_mol(derived_mol, global_indices):
    atom_set = set(global_indices)
    ring_info = derived_mol.GetRingInfo()

    for idx in global_indices:
        atom = derived_mol.GetAtomWithIdx(idx)
        if atom.IsInRing():
            # include every atom in every ring this atom belongs to
            for ring in ring_info.AtomRings():
                if idx in ring:
                    atom_set.update(ring)
        else:
            for nbr in atom.GetNeighbors():
                atom_set.add(nbr.GetIdx())

    # one-bond expansion of the full set for valence capping
    for idx in list(atom_set):
        for nbr in derived_mol.GetAtomWithIdx(idx).GetNeighbors():
            atom_set.add(nbr.GetIdx())

    # build RWMol, cap with implicit H, return
    ...
```

**This is GAFF/GAFF2-specific behaviour.** If other force fields are ever supported (OPLS,
CGenFF), the expansion depth and ring-inclusion rules will differ.  Encapsulate the expansion
in a `MinimalMoleculeExpander` protocol so it can be swapped:

```python
class MinimalMoleculeExpander(Protocol):
    def expand(self, mol: Chem.Mol, seed_indices: frozenset[int]) -> frozenset[int]: ...

class GAFFMinimalMoleculeExpander:
    """Ring-aware one-bond expansion for GAFF2 atom type assignment."""
    def expand(self, mol, seed_indices): ...
```

Inject `expander` into `MinimalMoleculeStrategy`.

---

### 6.4 Unified library — caching and deduplication

#### The problem

Every `FragmentLibraryPipeline.build()` call re-runs acpype on every trimer, even if identical
trimers appeared in a previous library build. Similarly, `MinimalMoleculeStrategy` re-runs
acpype every call for the same missing bond environment.

acpype is the bottleneck (seconds per molecule). Deduplication is essential for interactive use
and iterative library building over overlapping monomer sets.

#### Proposed additions to `FragmentLibrary`

```python
@dataclass
class TrimerCacheEntry:
    fragment_records: tuple[ParameterRecord, ...]
    atom_metadata_slice: dict[str, dict[int, AtomMetadata]]

@dataclass
class MinimalMolCacheEntry:
    # keyed by (canonical_smiles, local_indices, parameter)
    value: float

@dataclass
class FragmentLibrary:
    records: tuple[ParameterRecord, ...]
    atom_metadata: dict[str, dict[int, AtomMetadata]]
    schema_version: str = "2"                              # NEW
    trimer_cache: dict[str, TrimerCacheEntry] = field(...)   # NEW canonical_smiles → entry
    minimal_mol_cache: dict[str, float] = field(...)         # NEW compound key → value
```

#### Trimer deduplication in `FragmentLibraryBuilder`

Before calling `TrimerParameterisationPipeline.run(trimer)`:
1. Compute canonical SMILES of `trimer.mol` via `Chem.MolToSmiles(Chem.RemoveAllHs(trimer.mol))`.
2. Look up in `library.trimer_cache`.
3. If hit: pull records directly from cache; skip acpype entirely.
4. If miss: run pipeline, extract records, store in cache, proceed.

This makes the second library build for any overlapping monomer set near-instant.

#### Minimal molecule caching — where to own it

`MinimalMoleculeStrategy` is currently pure: given a polymer mol and indices, it constructs the
fragment, runs acpype, and returns a parameter value. The caching logic must know both the
construction (to build the cache key) and the library (to read/write).

**Recommended pattern — caching wrapper (no coupling between strategy and library):**

```python
class CachingMinimalMoleculeStrategy:
    _inner: MinimalMoleculeStrategy
    _library: FragmentLibrary           # mutable reference; written on cache miss

    def resolve(self, global_indices, parameter, context):
        mol, old_to_new = MinimalMoleculeStrategy._extract_minimal_mol(
            context.derived_mol, global_indices
        )
        local_indices = tuple(old_to_new[i] for i in global_indices)
        cache_key = _minimal_mol_key(mol, local_indices, parameter)

        cached = self._library.query_minimal_mol(cache_key)
        if cached is not None:
            return cached

        value = self._inner.resolve(global_indices, parameter, context)
        self._library.store_minimal_mol(cache_key, value)
        return value

def _minimal_mol_key(mol, local_indices, parameter):
    canonical = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
    return f"{canonical}|{local_indices}|{type(parameter).__name__}.{parameter.name}"
```

Advantages:
- `MinimalMoleculeStrategy` stays pure and independently testable.
- `FragmentLibrary` does not know about strategies.
- The wrapper is opt-in: users who don't want persistent caching use the bare strategy.

#### Searching the cache correctly

The cache key must be deterministic for the same chemical environment. Canonical SMILES from
RDKit (`Chem.MolToSmiles`) is canonical by definition — the same atoms in any connectivity order
produce the same string. Combined with local indices (which are DFS-ordered and therefore also
deterministic given the same canonical SMARTS pattern) and the parameter name, the key uniquely
identifies a computation.

#### Library merge for incremental builds

When a user builds a second library for new monomers that share some with the first:

```python
FragmentLibrary.merge(a: FragmentLibrary, b: FragmentLibrary) -> FragmentLibrary
```

- Union all `records` (deduplicate by `(pattern, local_indices, parameter)`).
- Union `atom_metadata`.
- Union `trimer_cache` (later build wins on conflict — trimers are deterministic).
- Union `minimal_mol_cache`.

---

### 6.5 Library schema versioning

Add `schema_version: str` field to `FragmentLibrary`. Current serialised format = `"1"`.
After dihedral multi-periodicity changes = `"2"`.

`FragmentLibrary.load(path)` raises `ValueError` if version is unrecognised.
Provide a `migrate_v1_to_v2(path)` utility that re-extracts dihedral terms from the stored
SMARTS hits (or simply deletes dihedral records so they are recomputed on next use).

---

### 6.6 1-4 scaling in ITP output

GAFF2 requires `fudgeQQ = 0.8333` and `fudgeLJ = 0.5` (equivalently `scee=1.2`, `scnb=2.0`)
in the GROMACS `[defaults]` section. These are currently absent from the ITP/TOP files.

`PolymerParameterisationPipeline._save_gromacs` should write a `[defaults]` section header
(or verify that parmed's GROMACS writer already inserts it — check `structure.defaults`).

---

## 7. Implementation Order

| Priority | Task | Effort | Blocking |
|---|---|---|---|
| 1 | Ring-aware minimal molecule expansion (`MinimalMoleculeExpander`) | Small | Nothing |
| 2 | Dihedral multi-periodicity (storage, extractor, tiler, schema v2) | Large | — |
| 3 | Improper dihedral extraction and tiling | Medium | Needs `AnnotatedImproper` |
| 4 | Minimal molecule caching wrapper | Small | Needs ring expansion (#1) |
| 5 | Trimer deduplication in library builder | Small | — |
| 6 | Library merge | Small | Needs trimer dedup (#5) |
| 7 | Schema versioning + v1→v2 migration | Small | Needs schema v2 (#2) |
| 8 | 1-4 scaling in ITP output | Small | — |

---

## 8. Notes for New Contributors

- **Run tests**: `uv run pytest` (335 tests, ~1.3 s). Coverage: `uv run pytest --cov=src/polymer_md`.
- **Demo scripts**: `demo_fragment_library_pipeline.py`, `demo_polymer_pipeline.py`, `demo_analysis.py`
  in the repo root. Run in order; each depends on output of the previous.
- **Never access parmed types directly.** Use `ParmedTypeResolver` in `utils/parmed_helper.py`.
  `bond.type` can be `None`; `dihedral.type` can be a `DihedralTypeList`. Both are handled there.
- **`derived_mol` is the ground truth.** All SMARTS matching in the tiler and analysis runs
  against `derived_mol` (derived from the parmed structure). Index `i` in `derived_mol` equals
  `structure.atoms[i]`.
- **Cap atoms are in the library (after the 2026-04-26 fix).** `TerminalFragmentExtractor` now
  extracts cap atoms as their own region. Re-build any library created before this date.
- **`polymer_atom_metadata` covers monomer heavy atoms only** (explicitly skips caps and H).
  The tiler logs a WARNING for atoms not in the library and not in metadata; this is expected
  for cap atoms if the library was built before the cap-coverage fix.
- **Strategies are injected per type.** Pass
  `missing_strategies={AtomParameter: ResiduePositionStrategy()}` to the tiler or pipeline.
  `StrictMissingParameterStrategy` (crash on miss) is the default for all types.
