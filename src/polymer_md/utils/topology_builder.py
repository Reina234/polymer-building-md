from __future__ import annotations

import parmed as pmd
from rdkit import Chem
from rdkit.Chem import Descriptors


class TopologyBuilder:
    @staticmethod
    def build(mol: Chem.Mol) -> pmd.Structure:
        structure = pmd.Structure()
        atoms = TopologyBuilder._add_atoms(mol, structure)
        TopologyBuilder._add_bonds(mol, atoms, structure)
        TopologyBuilder._add_angles(mol, atoms, structure)
        TopologyBuilder._add_dihedrals(mol, atoms, structure)
        TopologyBuilder._add_impropers(mol, atoms, structure)
        return structure

    @staticmethod
    def _add_atoms(mol: Chem.Mol, structure: pmd.Structure) -> list[pmd.Atom]:
        pt = Chem.GetPeriodicTable()
        has_conformer = mol.GetNumConformers() > 0
        conformer = mol.GetConformer() if has_conformer else None
        atoms = []
        for rdkit_atom in mol.GetAtoms():
            atomic_num = rdkit_atom.GetAtomicNum()
            atom = pmd.Atom(
                name=f"{rdkit_atom.GetSymbol()}{rdkit_atom.GetIdx()}",
                atomic_number=atomic_num,
                mass=pt.GetAtomicWeight(atomic_num),
            )
            if conformer is not None:
                pos = conformer.GetAtomPosition(rdkit_atom.GetIdx())
                atom.xx = pos.x
                atom.xy = pos.y
                atom.xz = pos.z
            structure.add_atom(atom, "MOL", 1)
            atoms.append(atom)
        return atoms

    @staticmethod
    def _add_bonds(mol: Chem.Mol, atoms: list[pmd.Atom], structure: pmd.Structure) -> None:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            structure.bonds.append(pmd.Bond(atoms[i], atoms[j]))

    @staticmethod
    def _add_angles(mol: Chem.Mol, atoms: list[pmd.Atom], structure: pmd.Structure) -> None:
        seen: set[tuple[int, int, int]] = set()
        for atom in mol.GetAtoms():
            j = atom.GetIdx()
            neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
            for pos_a, i in enumerate(neighbors):
                for k in neighbors[pos_a + 1:]:
                    key = (i, j, k)
                    if key not in seen:
                        seen.add(key)
                        structure.angles.append(pmd.Angle(atoms[i], atoms[j], atoms[k]))

    @staticmethod
    def _add_dihedrals(mol: Chem.Mol, atoms: list[pmd.Atom], structure: pmd.Structure) -> None:
        seen: set[tuple[int, int, int, int]] = set()
        for bond in mol.GetBonds():
            j = bond.GetBeginAtomIdx()
            k = bond.GetEndAtomIdx()
            j_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
            k_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
            for i in j_neighbors:
                for l in k_neighbors:
                    if i == l:
                        continue
                    forward = (i, j, k, l)
                    reverse = (l, k, j, i)
                    key = min(forward, reverse)
                    if key not in seen:
                        seen.add(key)
                        structure.dihedrals.append(
                            pmd.Dihedral(atoms[key[0]], atoms[key[1]], atoms[key[2]], atoms[key[3]])
                        )

    @staticmethod
    def _add_impropers(mol: Chem.Mol, atoms: list[pmd.Atom], structure: pmd.Structure) -> None:
        sp2 = Chem.rdchem.HybridizationType.SP2
        for atom in mol.GetAtoms():
            if atom.GetHybridization() != sp2:
                continue
            center = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) < 3:
                continue
            i, j, l = neighbors[0], neighbors[1], neighbors[2]
            structure.dihedrals.append(
                pmd.Dihedral(atoms[i], atoms[j], atoms[center], atoms[l], improper=True)
            )
