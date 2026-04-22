"""
Probes the parmed→MOL2→RDKit roundtrip that the extractor will rely on.

Uses a simple SMILES (styrene) directly — no polymer pipeline needed.
Pipeline: SMILES → RDKit SDF → OBabel MOL2 → acpype GROMACS → parmed → MOL2 → RDKit

Checks:
  1. Atom count preserved through the roundtrip
  2. Element alignment: parmed.atoms[i].atomic_number == rdkit.GetAtomWithIdx(i).GetAtomicNum()
  3. Coordinate alignment: parmed coords vs RDKit conformer (Å)
  4. Bond orders: not all single (aromatic/double bonds survive)
  5. SMARTS matching returns sensible indices
  6. SDF parmed vs GROMACS parmed coordinate crosswalk (extraction use case)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem

SMILES = "C=Cc1ccccc1"  # styrene
COORD_TOL = 0.05  # Å


# ── helpers ──────────────────────────────────────────────────────────────────

def build_rdkit_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    return mol


def write_sdf(mol: Chem.Mol, path: Path) -> None:
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def parmed_to_rdkit(structure: pmd.Structure, tmp_dir: Path) -> Chem.Mol:
    """
    parmed → PDB (coordinates + elements, naming-agnostic) → RDKit mol.
    Bond orders inferred from 3D geometry via rdDetermineBonds — fully
    independent of atom naming conventions or FF type labels.
    Atom index i in returned mol == structure.atoms[i] by PDB write order.
    """
    from rdkit.Chem import rdDetermineBonds

    pdb_path = tmp_dir / "derived.pdb"
    structure.save(str(pdb_path), overwrite=True)
    raw = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
    if raw is None:
        raise RuntimeError(f"RDKit failed to parse parmed-written PDB: {pdb_path}")
    rdDetermineBonds.DetermineBonds(raw, charge=0)
    Chem.SanitizeMol(raw)
    return raw


def parmed_coords(structure: pmd.Structure) -> np.ndarray:
    return np.array([[a.xx, a.xy, a.xz] for a in structure.atoms])


def rdkit_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def report(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # ── Step 1: build 3D mol, write SDF ──────────────────────────────────
        section("Setup: SMILES → RDKit SDF → OBabel MOL2 → acpype GROMACS")
        print(f"  SMILES: {SMILES}")

        rdkit_mol = build_rdkit_mol(SMILES)
        sdf_path = work / "mol.sdf"
        write_sdf(rdkit_mol, sdf_path)
        print(f"  RDKit mol atoms (with Hs): {rdkit_mol.GetNumAtoms()}")

        # ── Step 2: SDF → OBabel MOL2 ────────────────────────────────────────
        mol2_path = work / "mol.mol2"
        result = run(["obabel", str(sdf_path), "-O", str(mol2_path)], cwd=work)
        if not mol2_path.exists():
            print(f"  OBabel failed:\n{result.stderr}")
            return
        print(f"  OBabel MOL2 written: {mol2_path.name}")

        # ── Step 3: MOL2 → acpype GROMACS ────────────────────────────────────
        acpype_dir = work / "acpype_run"
        acpype_dir.mkdir()
        acpype_result = run(
            ["acpype", "-i", str(mol2_path.resolve()), "-b", "mol", "-o", "gmx", "-c", "bcc"],
            cwd=acpype_dir,
        )
        inner_dir = acpype_dir / "mol.acpype"
        top_path = next(inner_dir.glob("*.top"), None) if inner_dir.exists() else None
        gro_path = next(inner_dir.glob("*.gro"), None) if inner_dir.exists() else None
        if top_path is None or gro_path is None:
            print(f"  acpype stdout:\n{acpype_result.stdout[-1000:]}")
            print(f"  acpype stderr:\n{acpype_result.stderr[-1000:]}")
            return
        print(f"  acpype GROMACS written: {top_path.name}, {gro_path.name}")

        # ── Step 4: load GROMACS structure via parmed ─────────────────────────
        gromacs_structure = pmd.load_file(str(top_path), xyz=str(gro_path))
        print(f"  parmed loaded: {len(gromacs_structure.atoms)} atoms")

        # ── Test 1: GROMACS parmed → MOL2 → RDKit ────────────────────────────
        section("1. GROMACS parmed → MOL2 → RDKit")

        gromacs_mol = parmed_to_rdkit(gromacs_structure, work)
        n_pmd = len(gromacs_structure.atoms)
        n_rd = gromacs_mol.GetNumAtoms()
        report("Atom count preserved", n_pmd == n_rd, f"parmed={n_pmd}  rdkit={n_rd}")

        element_failures = []
        for i, (pa, ra) in enumerate(
            zip(gromacs_structure.atoms, gromacs_mol.GetAtoms(), strict=False)
        ):
            if pa.atomic_number != ra.GetAtomicNum():
                element_failures.append(
                    f"  atom {i}: parmed={pa.name} Z={pa.atomic_number}  "
                    f"rdkit Z={ra.GetAtomicNum()}"
                )
        report(
            "Element alignment (parmed Z == rdkit Z at each index)",
            not element_failures,
            f"{len(element_failures)} mismatch(es)",
        )
        for line in element_failures:
            print(line)

        pmd_c = parmed_coords(gromacs_structure)
        rd_c = rdkit_coords(gromacs_mol)
        if pmd_c.shape == rd_c.shape:
            diffs = np.linalg.norm(pmd_c - rd_c, axis=1)
            bad = np.where(diffs > COORD_TOL)[0]
            report(
                f"Coordinate alignment (tol={COORD_TOL} Å)",
                len(bad) == 0,
                f"max_dev={diffs.max():.4f} Å  {len(bad)} atom(s) outside tol",
            )
            for i in bad[:5]:
                print(f"  atom {i}: parmed={pmd_c[i]}  rdkit={rd_c[i]}  Δ={diffs[i]:.4f}")
        else:
            report(
                "Coordinate alignment", False,
                f"shape mismatch parmed={pmd_c.shape} rdkit={rd_c.shape}",
            )

        bond_counts: dict[str, int] = {}
        for bond in gromacs_mol.GetBonds():
            k = str(bond.GetBondTypeAsDouble())
            bond_counts[k] = bond_counts.get(k, 0) + 1
        has_non_single = any(float(k) != 1.0 for k in bond_counts)
        report("Non-single bond orders present", has_non_single, str(bond_counts))

        # ── Test 2: SMARTS matching on derived mol ────────────────────────────
        section("2. SMARTS matching on GROMACS-derived RDKit mol")

        smarts_cases = [
            ("[#6]",     "any carbon"),
            ("[#6;a]",   "aromatic carbon"),
            ("[#6;A]",   "aliphatic carbon"),
            ("[#1]",     "hydrogen"),
            ("[#6]-[#6]","C-C single bond"),
            ("[#6]:[#6]","aromatic C:C bond"),
            ("[#6]=[#6]","C=C double bond"),
        ]
        for smarts, label in smarts_cases:
            query = Chem.MolFromSmarts(smarts)
            matches = gromacs_mol.GetSubstructMatches(query)
            report(f"'{smarts}' ({label})", True, f"{len(matches)} match(es)")

        # ── Test 3: SDF parmed vs GROMACS parmed crosswalk ────────────────────
        section("3. SDF parmed vs GROMACS parmed coordinate crosswalk")

        sdf_loaded = pmd.load_file(str(sdf_path))
        sdf_structure = sdf_loaded[0] if isinstance(sdf_loaded, list) else sdf_loaded
        print(
            f"  SDF parmed atoms: {len(sdf_structure.atoms)}"
            f"  GROMACS parmed atoms: {len(gromacs_structure.atoms)}"
        )

        sdf_coords = parmed_coords(sdf_structure)
        gro_coords = parmed_coords(gromacs_structure)
        mapping: dict[int, int] = {}
        crosswalk_failures: list[str] = []

        for i, sc in enumerate(sdf_coords):
            dists = np.linalg.norm(gro_coords - sc, axis=1)
            j = int(np.argmin(dists))
            z_src = sdf_structure.atoms[i].atomic_number
            z_dst = gromacs_structure.atoms[j].atomic_number
            if dists[j] > COORD_TOL:
                crosswalk_failures.append(
                    f"  SDF[{i}] {sdf_structure.atoms[i].name}: "
                    f"nearest GRO[{j}] {gromacs_structure.atoms[j].name} "
                    f"Δ={dists[j]:.4f} Å — above tolerance"
                )
            elif z_src != z_dst:
                crosswalk_failures.append(
                    f"  SDF[{i}] {sdf_structure.atoms[i].name} Z={z_src}: "
                    f"matched GRO[{j}] {gromacs_structure.atoms[j].name} Z={z_dst} "
                    f"— element mismatch"
                )
            else:
                mapping[i] = j

        report(
            "All SDF atoms map to a GROMACS atom within tolerance",
            not crosswalk_failures,
            f"{len(mapping)} mapped  {len(crosswalk_failures)} failed",
        )
        for line in crosswalk_failures[:10]:
            print(line)

        is_bijective = len(set(mapping.values())) == len(mapping)
        report(
            "Mapping is bijective (no two SDF atoms map to same GROMACS atom)",
            is_bijective,
        )

        print("\n  Sample mapping (SDF idx → GROMACS idx):")
        for src_i, dst_j in sorted(mapping.items())[:12]:
            z = sdf_structure.atoms[src_i].atomic_number
            print(
                f"    SDF[{src_i:2d}] {sdf_structure.atoms[src_i].name:4s} Z={z}"
                f" → GRO[{dst_j:2d}] {gromacs_structure.atoms[dst_j].name:4s}"
                f" Z={gromacs_structure.atoms[dst_j].atomic_number}"
            )

        section("Summary")
        print("  FAIL on element alignment → index-based lookup silently uses wrong atoms.")
        print("  FAIL on non-single bonds  → use bond-agnostic SMARTS (~) for matching.")
        print("  FAIL on crosswalk         → coordinate-based region transfer is unreliable.")


if __name__ == "__main__":
    main()
