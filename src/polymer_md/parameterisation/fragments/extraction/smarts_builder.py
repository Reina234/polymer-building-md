from __future__ import annotations

from rdkit import Chem


class SmartsBuilder:
    @staticmethod
    def atom(atom: Chem.Atom) -> str:
        atomic_number = atom.GetAtomicNum()
        aromaticity_flag = "a" if atom.GetIsAromatic() else "A"
        formal_charge = atom.GetFormalCharge()
        if formal_charge != 0:
            charge_sign = "+" if formal_charge > 0 else ""
            return f"[#{atomic_number};{aromaticity_flag};{charge_sign}{formal_charge}]"
        return f"[#{atomic_number};{aromaticity_flag}]"

    @staticmethod
    def bond(bond: Chem.Bond) -> str:
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            return "-"
        if bond_type == Chem.BondType.DOUBLE:
            return "="
        if bond_type == Chem.BondType.AROMATIC:
            return ":"
        if bond_type == Chem.BondType.TRIPLE:
            return "#"
        return "~"

    @staticmethod
    def subgraph(
        mol: Chem.Mol,
        atom_indices: tuple[int, ...],
    ) -> tuple[str, dict[int, int]]:
        """Build SMARTS for a connected subgraph with deterministic atom ordering.

        Returns (smarts, global_to_local) where global_to_local[global_idx] gives the
        0-based position of that atom in the DFS traversal, which equals its query-atom
        index in the returned SMARTS (i.e. its position in GetSubstructMatches tuples).
        """
        if not atom_indices:
            return "", {}

        atom_set = frozenset(atom_indices)
        start = min(atom_indices)

        parent_of: dict[int, int | None] = {}
        back_edges: list[tuple[int, int]] = []
        seen_back_edge_bonds: set[frozenset[int]] = set()

        def classify(idx: int, par: int | None) -> None:
            parent_of[idx] = par
            for nbr_idx in sorted(
                n.GetIdx()
                for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetIdx() in atom_set
            ):
                if nbr_idx == par:
                    continue
                if nbr_idx in parent_of:
                    bond_key = frozenset({idx, nbr_idx})
                    if bond_key not in seen_back_edge_bonds:
                        seen_back_edge_bonds.add(bond_key)
                        back_edges.append((idx, nbr_idx))
                else:
                    classify(nbr_idx, idx)

        classify(start, None)

        closure_suffix: dict[int, list[str]] = {idx: [] for idx in atom_set}
        for closure_number, (from_idx, to_idx) in enumerate(back_edges, start=1):
            bond_smarts = SmartsBuilder.bond(mol.GetBondBetweenAtoms(from_idx, to_idx))
            closure_suffix[from_idx].append(f"{bond_smarts}{closure_number}")
            closure_suffix[to_idx].append(f"{bond_smarts}{closure_number}")

        global_to_local: dict[int, int] = {}
        counter = [0]

        def build(idx: int, par: int | None) -> str:
            global_to_local[idx] = counter[0]
            counter[0] += 1

            atom_smarts = SmartsBuilder.atom(mol.GetAtomWithIdx(idx))
            suffix = "".join(closure_suffix[idx])

            children = sorted(
                nbr_idx
                for nbr_idx in (
                    n.GetIdx()
                    for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if n.GetIdx() in atom_set
                )
                if nbr_idx != par and parent_of.get(nbr_idx) == idx
            )

            result = atom_smarts + suffix
            for position, child_idx in enumerate(children):
                bond_smarts = SmartsBuilder.bond(mol.GetBondBetweenAtoms(idx, child_idx))
                child_str = build(child_idx, idx)
                if position < len(children) - 1:
                    result += f"({bond_smarts}{child_str})"
                else:
                    result += f"{bond_smarts}{child_str}"

            return result

        smarts = build(start, None)
        return smarts, global_to_local
