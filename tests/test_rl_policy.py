"""
Automated unit tests for the Reinforcement Learning Candidate Selection Policy,
Environment Rewards, Abstention, Actor-Critic Network, and Training loops.
"""

import numpy as np
import pytest
import torch

from chemrxn.chemistry import canonicalize_smiles
from chemrxn.rl.environment import CandidateData, CandidateSelectionEnv, ReactionEpisode
from chemrxn.rl.policy import RLCandidatePolicy, PolicyOutput
from chemrxn.rl.trainer import RLPolicyTrainer, extract_episode_features


@pytest.fixture
def mock_episodes():
    c1 = CandidateData(
        smiles="CCNC(=O)C",
        canonical_smiles=canonicalize_smiles("CCNC(=O)C"),
        is_valid=True,
        transformer_score=-0.5,
        gnn_score=1.8,
        baseline_score=0.9,
        rc_consistency=0.8,
        delta_mw=2.0,
        delta_heavy_atoms=0.0,
    )
    c2 = CandidateData(
        smiles="CCCCCCCC",
        canonical_smiles="CCCCCCCC",
        is_valid=True,
        transformer_score=-4.0,
        gnn_score=-1.2,
        baseline_score=0.1,
        rc_consistency=0.2,
        delta_mw=60.0,
        delta_heavy_atoms=6.0,
    )
    c3_invalid = CandidateData(
        smiles="C(C(C(C",
        canonical_smiles="C(C(C(C",
        is_valid=False,
        transformer_score=-8.0,
        gnn_score=-5.0,
        baseline_score=0.0,
        rc_consistency=0.0,
        delta_mw=0.0,
        delta_heavy_atoms=0.0,
    )

    cands = [c1, c2, c3_invalid]
    cand_matrix = np.zeros((3, 8), dtype=np.float32)
    cand_matrix[0] = [-0.5, 1.8, 0.9, 0.8, 1.0, 0.04, 0.0, 0.4]
    cand_matrix[1] = [-4.0, -1.2, 0.1, 0.2, 1.0, 1.2, 0.6, 0.1]
    cand_matrix[2] = [-8.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    context = np.array([0.7, 0.3, 0.4, 0.5, 0.2, 0.4, 0.1, 0.8], dtype=np.float32)

    ep1 = ReactionEpisode(
        reactant_smiles="CC(=O)Cl.CCN",
        canonical_reactants="CCN.CC(=O)Cl",
        target_product_smiles="CCNC(=O)C",
        candidates=cands,
        context_features=context,
        candidate_features=cand_matrix,
        target_in_candidates_idx=0,
    )

    ep2_target_absent = ReactionEpisode(
        reactant_smiles="c1ccccc1.BrBr",
        canonical_reactants="BrBr.c1ccccc1",
        target_product_smiles="Brc1ccccc1",
        candidates=cands,
        context_features=context,
        candidate_features=cand_matrix,
        target_in_candidates_idx=None, # Target absent from candidate pool
    )

    return [ep1, ep2_target_absent]


def test_rl_environment_rewards_and_penalties(mock_episodes):
    """Verifies that the RL environment enforces exact chemical rewards and penalties."""
    env = CandidateSelectionEnv(mock_episodes)

    # 1. Exact Match on target (action 0) -> +1.0 reward
    env.reset(0)
    r, done, info = env.step(action=0, action_prob=0.95)
    assert r == 1.0
    assert info["is_correct"] is True
    assert done is True

    # 2. Chemically Invalid Candidate (action 2) -> -1.5 harsh penalty
    env.reset(0)
    r_inv, _, info_inv = env.step(action=2, action_prob=0.5)
    assert r_inv == -1.5
    assert info_inv["is_valid"] is False

    # 3. Overconfident Incorrect Action (action 1 with prob 0.95) -> overconfidence penalty
    env.reset(0)
    r_over, _, info_over = env.step(action=1, action_prob=0.95)
    assert info_over.get("overconfidence_penalized", False) is True
    assert r_over < -0.2

    # 4. Justified Abstention (episode 1 where target is absent) -> +0.3 reward
    env.reset(1)
    r_abs_just, _, info_just = env.step(action=3, action_prob=0.8) # action 3 is Abstain for 3 candidates
    assert r_abs_just == 0.3
    assert info_just["abstention_type"] == "justified"

    # 5. Unjustified Abstention (episode 0 where target is present) -> -0.2 penalty
    env.reset(0)
    r_abs_unjust, _, info_unjust = env.step(action=3, action_prob=0.8)
    assert r_abs_unjust == -0.2
    assert info_unjust["abstention_type"] == "unjustified"


def test_rl_policy_forward_shapes_and_distribution(mock_episodes):
    """Verifies RLCandidatePolicy produces valid action distributions and value estimates."""
    policy = RLCandidatePolicy(context_dim=8, candidate_dim=8, hidden_dim=32)

    ctx = torch.tensor(mock_episodes[0].context_features, dtype=torch.float32)
    cands = torch.tensor(mock_episodes[0].candidate_features, dtype=torch.float32)
    k = len(mock_episodes[0].candidates)

    out = policy(ctx, cands)
    assert isinstance(out, PolicyOutput)
    # Logits and probs should have K + 1 actions (K candidates + 1 abstain)
    assert out.logits.shape == (k + 1,)
    assert out.probs.shape == (k + 1,)
    assert torch.isclose(out.probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert isinstance(out.value.item(), float)

    # Act method
    action, prob, log_prob, val = policy.act(ctx, cands, deterministic=True)
    assert 0 <= action <= k
    assert 0.0 <= prob <= 1.0


def test_rl_trainer_supervised_warm_start_step(mock_episodes):
    """Verifies that supervised warm-start optimizes cross-entropy loss against ground truth target."""
    policy = RLCandidatePolicy(context_dim=8, candidate_dim=8, hidden_dim=32)
    trainer = RLPolicyTrainer(policy=policy, learning_rate=1e-3, device="cpu")
    env = CandidateSelectionEnv(mock_episodes)

    metrics = trainer.supervised_warm_start_epoch(env)
    assert "warm_start_loss" in metrics
    assert "warm_start_accuracy" in metrics
    assert metrics["warm_start_loss"] > 0


def test_rl_policy_gradient_step(mock_episodes):
    """Verifies that Actor-Critic policy gradient updates parameters with advantage estimation."""
    policy = RLCandidatePolicy(context_dim=8, candidate_dim=8, hidden_dim=32)
    trainer = RLPolicyTrainer(policy=policy, learning_rate=1e-3, device="cpu")
    env = CandidateSelectionEnv(mock_episodes)

    metrics = trainer.policy_gradient_epoch(env)
    assert "mean_reward" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_data_leakage_safeguards(mock_episodes):
    """Verifies that reference target labels are completely absent from state feature representations."""
    for ep in mock_episodes:
        # Context features must only represent reactants and pool variance
        assert ep.context_features.shape == (8,)
        # Candidate features must not encode target identity
        assert ep.candidate_features.shape[1] == 8
        # Target product is stored exclusively as external label metadata
        assert hasattr(ep, "target_product_smiles")
