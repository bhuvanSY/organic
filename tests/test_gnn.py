"""
Comprehensive unit tests for the Molecular Graph Neural Network (MPNN),
RDKit graph featurization, reaction center detection, batching, and candidate re-ranking.
"""

import pytest
import torch
from rdkit import Chem

from chemrxn.chemistry.graphs import (
    MolecularGraph,
    BatchMolecularGraph,
    mol_to_graph,
    batch_molecular_graphs,
    detect_reaction_centers,
    extract_atom_features,
    extract_bond_features,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)
from chemrxn.models.molecular_gnn import (
    MolecularReactionGNN,
    MPNNLayer,
    ReactionCenterHead,
    ReactionPairScorer,
)
from chemrxn.training.gnn_trainer import (
    GNNReactionTrainer,
    collate_gnn_reaction_batch,
    GNNReactionBatch,
)
from chemrxn.inference.candidate_ranker import (
    HybridCandidateRanker,
    RankedCandidate,
)


def test_atom_and_bond_feature_dimensions():
    """Verifies that atom and bond feature extractors produce exact expected dimensions."""
    mol = Chem.MolFromSmiles("CC(=O)O")
    assert mol is not None

    atom = mol.GetAtomWithIdx(0)
    a_feats = extract_atom_features(atom)
    assert len(a_feats) == NODE_FEATURE_DIM, f"Expected {NODE_FEATURE_DIM}, got {len(a_feats)}"

    bond = mol.GetBondWithIdx(0)
    b_feats = extract_bond_features(bond)
    assert len(b_feats) == EDGE_FEATURE_DIM, f"Expected {EDGE_FEATURE_DIM}, got {len(b_feats)}"


def test_molecular_graph_construction_and_isolated_atoms():
    """Tests graph construction on standard, disconnected, and zero-bond molecules."""
    # 1. Standard Molecule
    g_acetic = mol_to_graph("CC(=O)O")
    assert g_acetic is not None
    assert g_acetic.num_nodes == 4
    # 3 bonds, bidirectional -> 6 directed edges
    assert g_acetic.num_edges == 6
    assert g_acetic.x.shape == (4, NODE_FEATURE_DIM)
    assert g_acetic.edge_index.shape == (2, 6)
    assert g_acetic.edge_attr.shape == (6, EDGE_FEATURE_DIM)

    # 2. Disconnected reactant mixture
    g_mixture = mol_to_graph("CC(=O)Cl.CCN")
    assert g_mixture is not None
    assert g_mixture.num_nodes == 4 + 3  # 7 atoms
    assert g_mixture.num_edges == 6 + 4  # 10 directed edges

    # 3. Isolated ion / zero bonds
    g_ion = mol_to_graph("[Cl-]")
    assert g_ion is not None
    assert g_ion.num_nodes == 1
    assert g_ion.num_edges == 0
    assert g_ion.edge_index.shape == (2, 0)
    assert g_ion.edge_attr.shape == (0, EDGE_FEATURE_DIM)


def test_batch_molecular_graphs_offsets():
    """Verifies that batching correctly shifts edge indices by node offsets."""
    g1 = mol_to_graph("CC")  # 2 nodes, 2 edges
    g2 = mol_to_graph("CCC")  # 3 nodes, 4 edges
    assert g1 is not None and g2 is not None

    bg = batch_molecular_graphs([g1, g2])
    assert bg.num_graphs == 2
    assert bg.x.shape[0] == 5
    assert bg.edge_index.shape[1] == 6

    # Verify batch tensor
    expected_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    assert torch.equal(bg.batch, expected_batch)

    # Verify edges for g2 are shifted by 2
    g2_edges = bg.edge_index[:, 2:]
    assert g2_edges.min().item() >= 2
    assert g2_edges.max().item() <= 4


def test_reaction_center_detection_mapped_and_unmapped():
    """Tests reaction center detection on mapped reaction and unmapped MCS fallback."""
    # Mapped ester hydrolysis: [CH3:1][C:2](=[O:3])[OH:4].[NH3:5]>>[CH3:1][C:2](=[O:3])[NH2:5]
    r_mapped = "[CH3:1][C:2](=[O:3])[OH:4].[NH3:5]"
    p_mapped = "[CH3:1][C:2](=[O:3])[NH2:5]"
    centers, is_mapped = detect_reaction_centers(r_mapped, p_mapped)
    assert is_mapped is True
    # In reactants: CH3 (0), C(=O) (1), =O (0), OH (1), NH3 (1)
    assert sum(centers) >= 2, "Reaction centers should be detected on reacting atoms"

    # Unmapped reaction fallback
    r_unmapped = "CC(=O)Cl.CCN"
    p_unmapped = "CCNC(=O)C"
    centers_unmapped, is_mapped_un = detect_reaction_centers(r_unmapped, p_unmapped)
    assert is_mapped_un is False
    assert len(centers_unmapped) == 7
    assert sum(centers_unmapped) > 0


def test_mpnn_layer_forward():
    """Tests MPNNLayer shape preservation and message passing."""
    hidden_dim = 32
    layer = MPNNLayer(hidden_dim=hidden_dim)

    # 3 nodes, 2 bidirectional edges (1-0 and 1-2)
    h = torch.randn(3, hidden_dim)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, hidden_dim)

    h_out = layer(h, edge_index, edge_attr)
    assert h_out.shape == (3, hidden_dim)
    assert not torch.allclose(h, h_out), "Message passing should update representations"


def test_molecular_reaction_gnn_architecture():
    """Verifies that MolecularReactionGNN initializes from scratch and computes forward passes."""
    model = MolecularReactionGNN(hidden_dim=64, num_layers=2)
    assert model.count_parameters() > 10_000

    g1 = mol_to_graph("CC(=O)O")
    g2 = mol_to_graph("CCN")
    bg = batch_molecular_graphs([g1, g2])

    h_nodes, h_graph = model.encode_graph(bg)
    assert h_nodes.shape == (bg.x.size(0), 64)
    assert h_graph.shape == (2, 64)

    rc_logits = model.predict_reaction_center_logits(bg)
    assert rc_logits.shape == (bg.x.size(0),)

    pair_scores = model.score_candidates(bg, bg)
    assert pair_scores.shape == (2,)


def test_gnn_trainer_step_and_overfit():
    """Verifies that GNN multi-task trainer decreases loss over small reaction batches."""
    model = MolecularReactionGNN(hidden_dim=32, num_layers=2)
    trainer = GNNReactionTrainer(model=model, learning_rate=1e-3, device="cpu", use_amp=False)

    sample_triplets = [
        ("CC(=O)Cl.CCN", "CCNC(=O)C", "c1ccccc1"),
        ("CC(=O)O.CCN", "CCNC(=O)C", "CCCC"),
    ]
    batch = collate_gnn_reaction_batch(sample_triplets)
    assert batch is not None

    initial_loss = trainer.train_epoch([batch])["loss"]
    for _ in range(20):
        metrics = trainer.train_epoch([batch])

    final_loss = metrics["loss"]
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"
    assert metrics["rank_accuracy"] >= 0.5


def test_candidate_ranking_interface():
    """Tests the CandidateRankingResult structure and scoring logic."""
    model = MolecularReactionGNN(hidden_dim=32, num_layers=2)
    g_r = mol_to_graph("CC(=O)Cl.CCN")
    g_pos = mol_to_graph("CCNC(=O)C")
    g_neg = mol_to_graph("CCCCCCCC")
    assert g_r is not None and g_pos is not None and g_neg is not None

    scores = model.score_candidate_pool(g_r, [g_pos, g_neg])
    assert scores.shape == (2,)
    assert isinstance(scores[0].item(), float)
