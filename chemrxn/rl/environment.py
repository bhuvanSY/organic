"""
Reproducible Candidate-Selection Environment for Reinforcement Learning.
Defines:
- State: Reactant context features, candidate feature matrix, model scores.
- Action: Discrete choice among K candidates, or action K for ABSTAIN.
- Reward: Reference-product correctness, invalid molecule penalty, overconfidence penalty,
  and justified vs unjustified abstention rewards.
- Documented reward-hacking safeguards.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from chemrxn.chemistry.fingerprints import compute_morgan_fingerprint, compute_tanimoto_similarity
from chemrxn.chemistry.smiles import canonicalize_smiles, is_valid_smiles, compute_molecular_descriptors


@dataclass
class CandidateData:
    smiles: str
    canonical_smiles: str
    is_valid: bool
    transformer_score: float     # Length-normalized log-probability
    gnn_score: float             # Graph compatibility score
    baseline_score: float        # Classical ML ranker score
    rc_consistency: float        # Reaction center consistency
    delta_mw: float              # Mass divergence from reactants
    delta_heavy_atoms: float     # Heavy atom count divergence


@dataclass
class ReactionEpisode:
    reactant_smiles: str
    canonical_reactants: str
    target_product_smiles: Optional[str]  # Ground truth reference (used strictly for reward evaluation, never in state!)
    candidates: List[CandidateData]
    context_features: np.ndarray          # Reactant-level features [D_ctx]
    candidate_features: np.ndarray        # Candidate-level feature matrix [K, D_cand]
    target_in_candidates_idx: Optional[int] # Index of target in candidates, or None


class CandidateSelectionEnv:
    """
    Candidate Selection Environment for contextual bandit / reinforcement learning.
    Reward Structure:
    1. Correct Candidate Selection (Exact Match): +1.0
    2. Incorrect Candidate Selection (Valid Molecule): -0.2 + 0.3 * Tanimoto(C, Y*) in [-0.2, +0.1]
    3. Invalid Candidate Selection (RDKit Sanitization Failure): -1.5 (Harsh penalty)
    4. Overconfidence Penalty: -0.5 * max(0, pi(a) - 0.7) when action is incorrect
    5. Justified Abstention (Target absent or all invalid): +0.3
    6. Unjustified Abstention (Target present among candidates): -0.2

    Reward Hacking Safeguards:
    - If abstention reward is too high, the policy learns trivial 100% abstention.
    - If invalid penalty is too weak, the policy guesses corrupted SMILES.
    - Here, justified abstention (+0.3) is lower than true discovery (+1.0), but higher than
      guessing incorrectly (-0.2 to -1.5).
    """
    def __init__(
        self,
        episodes: List[ReactionEpisode],
        r_correct: float = 1.0,
        r_invalid: float = -1.5,
        r_abstain_justified: float = 0.3,
        r_abstain_unjustified: float = -0.2,
        r_incorrect_base: float = -0.2,
        r_incorrect_sim_scale: float = 0.3,
    ):
        if not episodes:
            raise ValueError("Environment requires at least one episode.")
        self.episodes = episodes
        self.r_correct = r_correct
        self.r_invalid = r_invalid
        self.r_abstain_justified = r_abstain_justified
        self.r_abstain_unjustified = r_abstain_unjustified
        self.r_incorrect_base = r_incorrect_base
        self.r_incorrect_sim_scale = r_incorrect_sim_scale

        self.current_idx = 0
        self.num_episodes = len(episodes)

    def reset(self, episode_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Resets environment to a specified or next episode.
        Returns:
            (context_features, candidate_features, num_candidates)
        """
        if episode_idx is not None:
            self.current_idx = episode_idx % self.num_episodes
        else:
            self.current_idx = (self.current_idx + 1) % self.num_episodes

        ep = self.episodes[self.current_idx]
        return ep.context_features, ep.candidate_features, len(ep.candidates)

    def step(
        self,
        action: int,
        action_prob: float = 1.0,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Executes an action in the current episode.
        Action in [0, K-1]: select candidate index.
        Action == K: ABSTAIN.

        Returns:
            (reward, done, info_dict)
        """
        ep = self.episodes[self.current_idx]
        k = len(ep.candidates)
        target_canon = canonicalize_smiles(ep.target_product_smiles, remove_mapping=True) if ep.target_product_smiles else None
        target_in_candidates = (ep.target_in_candidates_idx is not None)

        reward = 0.0
        info: Dict[str, Any] = {
            "episode_idx": self.current_idx,
            "action": action,
            "action_prob": action_prob,
            "is_abstain": (action == k),
            "is_correct": False,
            "is_valid": True,
            "tanimoto": 0.0,
        }

        # Case 1: Policy decides to ABSTAIN (action == k)
        if action >= k:
            info["is_abstain"] = True
            all_invalid = all(not c.is_valid for c in ep.candidates)

            # Justified if target is not among the candidate pool or all candidates are invalid
            if (not target_in_candidates) or all_invalid:
                reward = self.r_abstain_justified
                info["abstention_type"] = "justified"
            else:
                reward = self.r_abstain_unjustified
                info["abstention_type"] = "unjustified"

            info["reward"] = reward
            return float(reward), True, info

        # Case 2: Policy selects a candidate C_a
        chosen_cand = ep.candidates[action]
        info["chosen_smiles"] = chosen_cand.canonical_smiles
        info["is_valid"] = chosen_cand.is_valid

        # Penalty for selecting chemically invalid molecule
        if not chosen_cand.is_valid:
            reward = self.r_invalid
            info["reward"] = reward
            return float(reward), True, info

        # Check exact match against target reference product
        cand_canon = canonicalize_smiles(chosen_cand.canonical_smiles, remove_mapping=True) or chosen_cand.canonical_smiles
        if target_canon and cand_canon == target_canon:
            reward = self.r_correct
            info["is_correct"] = True
        else:
            # Partial credit based on molecular similarity to reference target
            sim = 0.0
            if target_canon and chosen_cand.is_valid:
                sim = compute_tanimoto_similarity(chosen_cand.canonical_smiles, target_canon) or 0.0
            info["tanimoto"] = round(float(sim), 4)

            reward = self.r_incorrect_base + self.r_incorrect_sim_scale * sim

            # Overconfidence penalty if policy assigned high certainty to an incorrect prediction
            if action_prob > 0.70:
                reward -= 0.5 * (action_prob - 0.70)
                info["overconfidence_penalized"] = True

        info["reward"] = float(round(reward, 4))
        return float(round(reward, 4)), True, info
