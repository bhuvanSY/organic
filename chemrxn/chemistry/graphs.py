"""
Molecular graph construction, atom/bond featurization, reaction center detection,
and batched disjoint-union graph representations using RDKit and PyTorch.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFMCS

from chemrxn.chemistry.smiles import parse_mol_safely


# Allowed categories for one-hot encodings
ALLOWED_ATOMIC_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # +1 for other -> 13
ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5, 6]  # 7
ALLOWED_FORMAL_CHARGES = [-2, -1, 0, 1, 2]  # +1 for other -> 6
ALLOWED_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]  # +1 for other -> 6
ALLOWED_NUM_HS = [0, 1, 2, 3, 4]  # 5
ALLOWED_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]  # +1 for other -> 4

ALLOWED_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]  # +1 for other -> 5
ALLOWED_BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]  # +1 for other -> 5

NODE_FEATURE_DIM = len(ALLOWED_ATOMIC_NUMS) + 1 + len(ALLOWED_DEGREES) + len(ALLOWED_FORMAL_CHARGES) + 1 + len(ALLOWED_HYBRIDIZATIONS) + 1 + 1 + len(ALLOWED_NUM_HS) + 1 + len(ALLOWED_CHIRALITY) + 1 + 1
EDGE_FEATURE_DIM = len(ALLOWED_BOND_TYPES) + 1 + 1 + 1 + len(ALLOWED_BOND_STEREO) + 1


def _one_hot(val: Any, allowed: List[Any]) -> List[float]:
    """Helper to generate one-hot encoding with an 'other' bucket."""
    res = [0.0] * (len(allowed) + 1)
    if val in allowed:
        res[allowed.index(val)] = 1.0
    else:
        res[-1] = 1.0
    return res


def extract_atom_features(atom: Chem.Atom) -> List[float]:
    """Extracts a fixed-length feature vector for an RDKit Atom."""
    feats: List[float] = []
    # 1. Atomic Number (13)
    feats.extend(_one_hot(atom.GetAtomicNum(), ALLOWED_ATOMIC_NUMS))
    # 2. Degree (7)
    deg = atom.GetTotalDegree()
    deg_onehot = [1.0 if deg == d else 0.0 for d in ALLOWED_DEGREES]
    feats.extend(deg_onehot)
    # 3. Formal Charge (6)
    feats.extend(_one_hot(atom.GetFormalCharge(), ALLOWED_FORMAL_CHARGES))
    # 4. Hybridization (6)
    feats.extend(_one_hot(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS))
    # 5. Aromaticity (1)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    # 6. Total Hydrogens (5)
    num_h = min(atom.GetTotalNumHs(), 4)
    h_onehot = [1.0 if num_h == h else 0.0 for h in ALLOWED_NUM_HS]
    feats.extend(h_onehot)
    # 7. In Ring (1)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    # 8. Chirality (4)
    feats.extend(_one_hot(atom.GetChiralTag(), ALLOWED_CHIRALITY))
    # 9. Has Atom Map (1)
    feats.append(1.0 if atom.GetAtomMapNum() > 0 else 0.0)
    return feats


def extract_bond_features(bond: Chem.Bond) -> List[float]:
    """Extracts a fixed-length feature vector for an RDKit Bond."""
    feats: List[float] = []
    # 1. Bond Type (5)
    feats.extend(_one_hot(bond.GetBondType(), ALLOWED_BOND_TYPES))
    # 2. Conjugated (1)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    # 3. In Ring (1)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    # 4. Stereo (5)
    feats.extend(_one_hot(bond.GetStereo(), ALLOWED_BOND_STEREO))
    return feats


@dataclass
class MolecularGraph:
    """Single molecular graph with PyTorch tensors."""
    x: torch.Tensor             # [num_nodes, NODE_FEATURE_DIM]
    edge_index: torch.Tensor    # [2, num_edges] (bidirectional)
    edge_attr: torch.Tensor     # [num_edges, EDGE_FEATURE_DIM]
    num_nodes: int
    num_edges: int
    smiles: str = ""
    atom_map_nums: Optional[List[int]] = None

    def to(self, device: Union[str, torch.device]) -> "MolecularGraph":
        return MolecularGraph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            smiles=self.smiles,
            atom_map_nums=self.atom_map_nums,
        )


@dataclass
class BatchMolecularGraph:
    """
    Disjoint-union batched molecular graph representation for vectorized PyTorch operations.
    Avoids slow one-molecule-at-a-time GPU operations.
    """
    x: torch.Tensor             # [total_nodes, NODE_FEATURE_DIM]
    edge_index: torch.Tensor    # [2, total_edges]
    edge_attr: torch.Tensor     # [total_edges, EDGE_FEATURE_DIM]
    batch: torch.Tensor         # [total_nodes] mapping each node to graph index in [0, B-1]
    num_graphs: int
    num_nodes_per_graph: List[int]

    def to(self, device: Union[str, torch.device]) -> "BatchMolecularGraph":
        return BatchMolecularGraph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            batch=self.batch.to(device),
            num_graphs=self.num_graphs,
            num_nodes_per_graph=self.num_nodes_per_graph,
        )


def mol_to_graph(smiles_or_mol: Union[str, Chem.Mol]) -> Optional[MolecularGraph]:
    """
    Converts a SMILES string or RDKit Mol object into a MolecularGraph.
    Handles disconnected components (e.g. 'A.B.C') automatically as a single disjoint graph.
    """
    if isinstance(smiles_or_mol, str):
        mol = parse_mol_safely(smiles_or_mol)
        smiles = smiles_or_mol
    else:
        mol = smiles_or_mol
        smiles = Chem.MolToSmiles(mol) if mol is not None else ""

    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Node features
    node_features = []
    atom_maps = []
    for atom in mol.GetAtoms():
        node_features.append(extract_atom_features(atom))
        atom_maps.append(atom.GetAtomMapNum())

    x = torch.tensor(node_features, dtype=torch.float32)

    # Edge features & index (bidirectional: u->v and v->u)
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_feats: List[List[float]] = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        b_feat = extract_bond_features(bond)

        # u -> v
        edge_src.append(u)
        edge_dst.append(v)
        edge_feats.append(b_feat)

        # v -> u
        edge_src.append(v)
        edge_dst.append(u)
        edge_feats.append(b_feat)

    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    return MolecularGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_atoms,
        num_edges=edge_index.size(1),
        smiles=smiles,
        atom_map_nums=atom_maps,
    )


def batch_molecular_graphs(graphs: List[MolecularGraph]) -> BatchMolecularGraph:
    """
    Batches a list of MolecularGraph objects into a single disjoint-union BatchMolecularGraph.
    Shifted edge indices ensure isolated message passing between individual graphs.
    """
    if not graphs:
        raise ValueError("Cannot batch empty list of graphs")

    total_nodes = sum(g.num_nodes for g in graphs)
    total_edges = sum(g.num_edges for g in graphs)

    x_list = [g.x for g in graphs]
    edge_attr_list = [g.edge_attr for g in graphs]

    batch_list = []
    edge_index_list = []
    num_nodes_per_graph = []

    node_offset = 0
    for i, g in enumerate(graphs):
        num_nodes_per_graph.append(g.num_nodes)
        batch_list.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        if g.num_edges > 0:
            edge_index_list.append(g.edge_index + node_offset)
        node_offset += g.num_nodes

    x = torch.cat(x_list, dim=0)
    batch = torch.cat(batch_list, dim=0)

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    return BatchMolecularGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        num_graphs=len(graphs),
        num_nodes_per_graph=num_nodes_per_graph,
    )


def detect_reaction_centers(
    reactant_smiles: str,
    product_smiles: str,
    timeout_sec: int = 2,
) -> Tuple[List[int], bool]:
    """
    Detects reaction center atoms in the reactant molecule.
    Returns:
        (reaction_center_mask, is_mapped):
            reaction_center_mask: List of 0 or 1 for each atom in reactant_mol
            is_mapped: True if ground-truth atom mapping was used, False if fallback MCS was used.
    """
    r_mol = parse_mol_safely(reactant_smiles)
    p_mol = parse_mol_safely(product_smiles)

    if r_mol is None or p_mol is None:
        num_atoms = r_mol.GetNumAtoms() if r_mol is not None else 0
        return [0] * num_atoms, False

    num_r_atoms = r_mol.GetNumAtoms()
    r_maps = [atom.GetAtomMapNum() for atom in r_mol.GetAtoms()]
    p_maps = {atom.GetAtomMapNum(): atom for atom in p_mol.GetAtoms() if atom.GetAtomMapNum() > 0}

    # Case 1: Ground-truth Atom Mapping is present
    mapped_count = sum(1 for m in r_maps if m > 0 and m in p_maps)
    if mapped_count >= 2:
        centers = [0] * num_r_atoms
        for i, atom in enumerate(r_mol.GetAtoms()):
            m = atom.GetAtomMapNum()
            if m == 0 or m not in p_maps:
                # Atom lost or unmapped -> participating in reaction / leaving group
                centers[i] = 1
                continue

            p_atom = p_maps[m]
            # Check if degree, formal charge, or aromaticity changed
            if atom.GetDegree() != p_atom.GetDegree() or atom.GetFormalCharge() != p_atom.GetFormalCharge():
                centers[i] = 1
                continue

            # Check if neighboring bonds/atoms changed
            r_neighbors = {
                (nb.GetAtomMapNum(), bond.GetBondType())
                for nb, bond in [(b.GetOtherAtom(atom), b) for b in atom.GetBonds()]
            }
            p_neighbors = {
                (nb.GetAtomMapNum(), bond.GetBondType())
                for nb, bond in [(b.GetOtherAtom(p_atom), b) for b in p_atom.GetBonds()]
            }
            if r_neighbors != p_neighbors:
                centers[i] = 1

        return centers, True

    # Case 2: Unmapped or incomplete mapping -> Maximum Common Substructure (MCS) heuristic
    try:
        # For multi-component reactants, find the major reactant with largest atom count
        submols = Chem.GetMolFrags(r_mol, asMols=True)
        major_mol = max(submols, key=lambda m: m.GetNumAtoms()) if submols else r_mol

        mcs_res = rdFMCS.FindMCS(
            [major_mol, p_mol],
            timeout=timeout_sec,
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
        )

        centers = [0] * num_r_atoms
        if mcs_res and mcs_res.smartsString:
            mcs_pattern = Chem.MolFromSmarts(mcs_res.smartsString)
            if mcs_pattern:
                matches = r_mol.GetSubstructMatches(mcs_pattern)
                if matches:
                    matched_atoms = set(matches[0])
                    # Atoms NOT in the common core, or boundary atoms in the core, are reacting
                    for i in range(num_r_atoms):
                        if i not in matched_atoms:
                            centers[i] = 1
                        else:
                            # If matched atom has a neighbor outside matched set, it is a boundary/reacting atom
                            atom = r_mol.GetAtomWithIdx(i)
                            if any(nb.GetIdx() not in matched_atoms for nb in atom.GetNeighbors()):
                                centers[i] = 1
                    return centers, False
    except Exception:
        pass

    # Fallback: mark all 0
    return [0] * num_r_atoms, False
