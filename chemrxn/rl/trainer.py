"""
RL Policy Trainer implementing:
1. Feature extraction to convert raw reactions into reproducible ReactionEpisodes
2. Supervised warm-start pretraining
3. Policy-gradient optimization (Actor-Critic with learned baseline and entropy regularization)
4. Anti-reward-hacking metrics and training tracking
"""

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from chemrxn.chemistry.fingerprints import compute_morgan_fingerprint, compute_tanimoto_similarity
from chemrxn.chemistry.graphs import detect_reaction_centers, mol_to_graph
from chemrxn.chemistry.smiles import canonicalize_smiles, compute_molecular_descriptors, is_valid_smiles
from chemrxn.rl.environment import CandidateData, CandidateSelectionEnv, ReactionEpisode
from chemrxn.rl.policy import RLCandidatePolicy


def extract_episode_features(
    reactant_smiles: str,
    target_product_smiles: Optional[str],
    raw_candidates_with_scores: List[Tuple[str, float]], # List of (smiles, tf_log_prob)
    gnn_model: Any,
    baseline_ranker: Any,
    device: torch.device,
) -> Optional[ReactionEpisode]:
    """
    Constructs a single ReactionEpisode with isolated context and candidate features.
    The target product is stored STRICTLY for reward calculation, never in state features!
    """
    canon_r = canonicalize_smiles(reactant_smiles, remove_mapping=True) or reactant_smiles
    if not is_valid_smiles(canon_r):
        return None

    r_desc = compute_molecular_descriptors(canon_r) or {}
    r_mw = float(r_desc.get("molecular_weight", 150.0))
    r_logp = float(r_desc.get("logp", 1.5))
    r_tpsa = float(r_desc.get("tpsa", 30.0))
    r_heavy = float(r_desc.get("num_heavy_atoms", 10))
    r_h_donors = float(r_desc.get("num_h_donors", 1))
    r_h_acceptors = float(r_desc.get("num_h_acceptors", 2))

    r_fp = compute_morgan_fingerprint(canon_r)

    # GNN Reaction Centers on Reactants
    g_r = mol_to_graph(canon_r)
    atom_rc_probs: List[float] = []
    if g_r is not None and gnn_model is not None:
        from chemrxn.chemistry.graphs import batch_molecular_graphs
        batched_r = batch_molecular_graphs([g_r]).to(device)
        with torch.no_grad():
            rc_logits = gnn_model.predict_reaction_center_logits(batched_r)
            atom_rc_probs = torch.sigmoid(rc_logits).cpu().tolist()

    # GNN candidate scoring
    cand_graphs = []
    cand_graph_indices = []
    gnn_scores_map: Dict[str, float] = {}

    for idx, (c_smi, _) in enumerate(raw_candidates_with_scores):
        g = mol_to_graph(c_smi)
        if g is not None:
            cand_graphs.append(g)
            cand_graph_indices.append(c_smi)

    if cand_graphs and g_r is not None and gnn_model is not None:
        with torch.no_grad():
            scores_tensor = gnn_model.score_candidate_pool(g_r, cand_graphs)
            for s_smi, val in zip(cand_graph_indices, scores_tensor.cpu().tolist()):
                gnn_scores_map[s_smi] = float(val)

    # Baseline ranker scoring
    baseline_scores_map: Dict[str, float] = {}
    cand_strings = [c[0] for c in raw_candidates_with_scores]
    if baseline_ranker is not None:
        try:
            b_scored = baseline_ranker.score_candidates(canon_r, cand_strings)
            baseline_scores_map = {smi: score for smi, score in b_scored}
        except Exception:
            baseline_scores_map = {smi: 0.5 for smi in cand_strings}
    else:
        baseline_scores_map = {smi: 0.5 for smi in cand_strings}

    target_canon = canonicalize_smiles(target_product_smiles, remove_mapping=True) if target_product_smiles else None
    target_idx: Optional[int] = None

    candidate_data_list: List[CandidateData] = []
    cand_feature_rows: List[List[float]] = []

    for idx, (c_smi, tf_log_prob) in enumerate(raw_candidates_with_scores):
        c_canon = canonicalize_smiles(c_smi, remove_mapping=True) or c_smi
        valid = is_valid_smiles(c_canon)
        c_desc = compute_molecular_descriptors(c_canon) if valid else None

        c_mw = float(c_desc.get("molecular_weight", r_mw)) if c_desc else r_mw
        c_heavy = float(c_desc.get("num_heavy_atoms", r_heavy)) if c_desc else r_heavy

        # Normalized length log-probability
        norm_tf = float(tf_log_prob) / max(len(c_smi), 1)
        g_score = gnn_scores_map.get(c_smi, -5.0)
        b_score = baseline_scores_map.get(c_smi, 0.5)

        # Reaction center consistency
        rc_cons = 0.5
        if valid and atom_rc_probs:
            rc_atoms, _ = detect_reaction_centers(canon_r, c_canon)
            if rc_atoms and sum(rc_atoms) > 0:
                matched = [atom_rc_probs[i] for i, r in enumerate(rc_atoms) if r == 1 and i < len(atom_rc_probs)]
                if matched:
                    rc_cons = float(np.mean(matched))

        delta_mw = abs(c_mw - r_mw)
        delta_heavy = abs(c_heavy - r_heavy)

        # Tanimoto to reactants
        tan_r = 0.0
        if valid and r_fp is not None:
            c_fp = compute_morgan_fingerprint(c_canon)
            if c_fp is not None:
                tan_r = float(compute_tanimoto_similarity(canon_r, c_canon) or 0.0)

        # Check target index
        if target_canon and c_canon == target_canon and target_idx is None:
            target_idx = idx

        cand_data = CandidateData(
            smiles=c_smi,
            canonical_smiles=c_canon,
            is_valid=valid,
            transformer_score=norm_tf,
            gnn_score=g_score,
            baseline_score=b_score,
            rc_consistency=rc_cons,
            delta_mw=delta_mw,
            delta_heavy_atoms=delta_heavy,
        )
        candidate_data_list.append(cand_data)

        # 8 candidate features: [norm_tf, gnn_score, baseline_score, rc_cons, valid, delta_mw, delta_heavy, tanimoto_to_r]
        row = [
            norm_tf,
            g_score,
            b_score,
            rc_cons,
            1.0 if valid else 0.0,
            delta_mw / 50.0,      # Normalized scale
            delta_heavy / 10.0,   # Normalized scale
            tan_r,
        ]
        cand_feature_rows.append(row)

    cand_matrix = np.array(cand_feature_rows, dtype=np.float32)

    # Compute candidate pool statistics for context: entropy & score variance
    tf_scores = np.array([c[0] for c in cand_feature_rows])
    score_var = float(np.var(tf_scores)) if len(tf_scores) > 1 else 0.0
    # Softmax entropy
    exp_s = np.exp(tf_scores - np.max(tf_scores))
    p_s = exp_s / np.sum(exp_s)
    pool_entropy = float(-np.sum(p_s * np.log(p_s + 1e-8)))

    # 8 context features: [mw, logp, tpsa, heavy, h_donors, h_acceptors, score_var, pool_entropy]
    context_vector = np.array([
        r_mw / 200.0,
        r_logp / 5.0,
        r_tpsa / 100.0,
        r_heavy / 20.0,
        r_h_donors / 5.0,
        r_h_acceptors / 5.0,
        score_var,
        pool_entropy,
    ], dtype=np.float32)

    return ReactionEpisode(
        reactant_smiles=reactant_smiles,
        canonical_reactants=canon_r,
        target_product_smiles=target_product_smiles,
        candidates=candidate_data_list,
        context_features=context_vector,
        candidate_features=cand_matrix,
        target_in_candidates_idx=target_idx,
    )


class RLPolicyTrainer:
    """
    Manages supervised warm-start pretraining and policy-gradient fine-tuning for RLCandidatePolicy.
    """
    def __init__(
        self,
        policy: RLCandidatePolicy,
        learning_rate: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip

        self.optimizer = AdamW(self.policy.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.cross_entropy = nn.CrossEntropyLoss()

    def supervised_warm_start_epoch(self, env: CandidateSelectionEnv) -> Dict[str, float]:
        """
        Runs one supervised pretraining epoch on environment episodes.
        Target:
            - If target is present at candidate k -> class k
            - If target is absent or all invalid -> class K (Abstain)
        """
        self.policy.train()
        total_loss = 0.0
        correct_actions = 0
        n_episodes = env.num_episodes

        for ep_idx in range(n_episodes):
            ctx, cands, k = env.reset(ep_idx)
            ep = env.episodes[ep_idx]

            # Determine supervision label
            if ep.target_in_candidates_idx is not None:
                target_action = ep.target_in_candidates_idx
            else:
                target_action = k  # Abstain

            ctx_t = torch.tensor(ctx, dtype=torch.float32, device=self.device)
            cands_t = torch.tensor(cands, dtype=torch.float32, device=self.device)
            target_t = torch.tensor(target_action, dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            out = self.policy.forward(ctx_t, cands_t)
            loss = self.cross_entropy(out.logits.unsqueeze(0), target_t.unsqueeze(0))

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += float(loss.item())
            pred_action = int(torch.argmax(out.probs).item())
            if pred_action == target_action:
                correct_actions += 1

        return {
            "warm_start_loss": round(total_loss / max(n_episodes, 1), 4),
            "warm_start_accuracy": round(correct_actions / max(n_episodes, 1), 4),
        }

    def policy_gradient_epoch(self, env: CandidateSelectionEnv) -> Dict[str, float]:
        """
        Runs one Actor-Critic Policy Gradient epoch across all episodes.
        Loss = -log_prob * Advantage + value_loss - entropy_bonus.
        """
        self.policy.train()
        total_reward = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_correct = 0
        num_invalid = 0
        num_abstained = 0
        n_episodes = env.num_episodes

        for ep_idx in range(n_episodes):
            ctx, cands, k = env.reset(ep_idx)

            ctx_t = torch.tensor(ctx, dtype=torch.float32, device=self.device)
            cands_t = torch.tensor(cands, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            out = self.policy.forward(ctx_t, cands_t)

            # Sample action from policy distribution
            action = int(out.dist.sample().item())
            log_prob = out.dist.log_prob(torch.tensor(action, device=self.device))
            action_prob = float(out.probs[action].item())
            value = out.value

            # Environment step
            reward, done, info = env.step(action, action_prob=action_prob)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)

            # Advantage = Reward - Value(s)
            advantage = reward_t - value.detach()

            # Actor & Critic losses
            policy_loss = -log_prob * advantage
            entropy_loss = -self.entropy_coef * out.dist.entropy()
            value_loss = self.value_coef * 0.5 * (reward_t - value).pow(2)

            total_step_loss = policy_loss + entropy_loss + value_loss

            total_step_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()

            total_reward += reward
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())

            if info.get("is_correct", False):
                num_correct += 1
            if not info.get("is_valid", True):
                num_invalid += 1
            if info.get("is_abstain", False):
                num_abstained += 1

        n = max(n_episodes, 1)
        return {
            "mean_reward": round(total_reward / n, 4),
            "policy_loss": round(total_policy_loss / n, 4),
            "value_loss": round(total_value_loss / n, 4),
            "accuracy": round(num_correct / n, 4),
            "invalid_rate": round(num_invalid / n, 4),
            "abstention_rate": round(num_abstained / n, 4),
        }
