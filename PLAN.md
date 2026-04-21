# polymer-md Architecture Plan

## End Goal

Fragment-based MD parameterisation of random copolymers:
1. Build a library of trimer force-field parameters (acpype/GAFF2)
2. Tile those parameters onto stochastically constructed full polymers
3. Analyse and compare parameterisation results across methods

---

## Dependency Flow (acyclic, direction fixed)

```
core  ←  geometry  ←  building  ←  conversion  ←  parameterisation  ←  analysis
```

---

## Pipeline 1 — Fragment Library Building (Phase 1 + 2)

```
MonomerSpec list
  → TransitionMatrix (Solver)
  → TrimerBuilder.build_all()  →  filter by p_min
  → For each TrimerResult:
      embed 3D (ConformerGenerator)  →  write SDF
      OBabel SDF → MOL2
      acpype MOL2 → GROMACS (ITP/GRO/TOP)
      ParameterisedTrimer(trimer_result, gromacs_files, structure)
  → InteriorFragmentExtractor  +  TerminalFragmentExtractor
  → FragmentLibraryBuilder
  → FragmentLibrary  (SMARTS-keyed, portable, serialisable)
```

## Pipeline 2 — Full Polymer Parameterisation (Phase 3, future)

```
FragmentLibrary + MonomerSpecs + n
  → RandomPolymerBuilder → Polymer
  → embed 3D → GRO → re-derive SMILES via parmed
  → SMARTS-match library → tile parameters
  → raise MissingParameterError on gaps
  → charge neutrality adjustment
  → output ITP
```

## Pipeline 3 — Analysis (Phase 4, future)

```
ExtractionSpec (SMARTS + param flags)  →  to_fragment()  →  Fragment
ParameterComparator.extract(structures)  →  AnalysisResult
ParameterComparator.compare(result_a, result_b)  →  ComparisonResult
```

Analysis is fully decoupled from building/geometry/extraction — usable standalone.

---

## File Structure

```
src/polymer_md/
├── core/                           (stable, no changes)
│
├── geometry/                       NEW
│   ├── base.py                     ConformerGenerator Protocol
│   ├── etkdg.py                    ETKDGConformerGenerator(use_uff, random_seed, max_attempts)
│   └── obabel_conformer.py         OBabelConformerGenerator(forcefield, steps)
│
├── building/
│   ├── data_models/
│   │   └── trimer.py               EXTEND: TrimerRegion enum, region atom index sets on TrimerResult
│   ├── solvers/
│   ├── monomer_converter.py        RENAMED from conversion.py
│   ├── growing_polymer.py
│   ├── random_polymer.py
│   └── trimer_builder.py           EXTEND: tag atoms by region during assembly
│
├── conversion/
│   ├── file_formats.py             ADD FileFormats.SDF
│   ├── obabel.py                   ADD SDF→MOL2 handler, REMOVE broken no-3D SMILES→MOL2
│   ├── acpype.py                   ADD charge_method: str = "bcc"
│   └── parmed.py
│
├── parameterisation/
│   ├── data_models/
│   │   ├── parameterised_mol.py
│   │   └── parameterised_trimer.py NEW
│   ├── fragments/
│   │   ├── data_models/
│   │   │   └── match.py            REVISE: ParameterRecord → smarts_pattern + local_indices
│   │   ├── extraction/             NEW
│   │   │   ├── interior.py         InteriorFragmentExtractor
│   │   │   └── terminal.py         TerminalFragmentExtractor
│   │   ├── matching/
│   │   └── library.py              REVISE: query_for_atoms() SMARTS-based + to_fragments()
│   └── pipeline.py                 NEW: TrimerParameterisationPipeline orchestrator
│
├── analysis/                       NEW stub (Phase 4)
│   ├── extraction_spec.py          ExtractionSpec(pattern, flags) → to_fragment()
│   ├── comparator.py               ParameterComparator
│   └── results.py                  ParameterDistribution, AnalysisResult, ComparisonResult
│
└── utils/
    └── rdkit_helper.py             ADD write_sdf, mol_to_pdb
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Cap default | `BuiltinCap.METHYL` | More realistic polymer-interior environment than H |
| Conformer default | `ETKDGConformerGenerator(use_uff=False)` | Better for larger molecules; UFF opt optional |
| Conformer location | Injected into pipeline, not on Polymer | Polymer is pure data; generator is a strategy |
| Charge method | `"bcc"` (AM1-BCC) default, configurable | GAFF2+AM1-BCC is literature standard |
| p_min default | `0.0` | Include all reachable trimers; user raises to reduce compute |
| MonomerComposition vs MonomerSpec | Keep separate | Solvers only need id+weight; MonomerSpec is user entry point |
| ParameterRecord key | `smarts_pattern + local_indices` | Global indices are molecule-specific; SMARTS is portable |
| Angle extraction scope | Middle atom must be central | Captures B2-B1-A1; excludes A-centric angles (those from A-central trimers) |
| OBabelConformerGenerator | Does NOT preserve atom tags | Converts via SMILES round-trip; use ETKDG for tagged mols |
| Analysis coupling | `analysis/` does NOT import `building/` or `geometry/` | Standalone reuse for comparing any parameterisation method |

## Configurability Principle

All pipeline parameters have sensible defaults but are fully overridable via constructor injection.
No hardcoded caps, solvers, conformer methods, charge methods, or strategies.
Minimum viable usage: `TrimerParameterisationPipeline(specs=[...]).run(output_dir)`.

---

## Phase Roadmap

| Phase | Status | Content |
|---|---|---|
| 0 | Done | Core building, solvers, trimer builder, conversion layer, fragment data models |
| 1 | Current | geometry/, TrimerResult region tags, TrimerParameterisationPipeline, main.py test |
| 2 | Next | ParameterRecord redesign, InteriorFragmentExtractor, TerminalFragmentExtractor, FragmentLibraryBuilder |
| 3 | Future | PolymerParameterisationTiler, charge neutrality, full polymer ITP |
| 4 | Future | analysis/ (ExtractionSpec, ParameterComparator, AnalysisResult) |
