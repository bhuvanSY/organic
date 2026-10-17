"""
Supervised Candidate-Ranking Baseline.
Scores and ranks plausible product candidates for a given reaction.
Takes (Reactants, Candidate Product) pairs and predicts probability of being the true reaction outcome.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np

from chemrxn.baselines.featurizer import ReactionFeaturizer
from chemrxn.chemistry.smiles import canonicalize_smiles


class SupervisedCandidateRanker:
    """
    Supervised ranker for multi-candidate product selection.
    Combines reactant and candidate molecular features to evaluate synthesis plausibility.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        seed: int = 42,
        fp_nbits: int = 1024,
    ):
        self.model_type = model_type.lower()
        self.seed = seed
        self.fp_nbits = fp_nbits
        self.featurizer = ReactionFeaturizer(fp_nbits=fp_nbits, include_descriptors=True)
        self.model = self._create_model()
        self.is_trained = False

    def _create_model(self):
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.seed,
                    eval_metric="logloss",
                    tree_method="hist",
                    device=device,
                    n_jobs=-1 if device == "cpu" else None,
                )
            except Exception:
                self.model_type = "hist_gb"

        if self.model_type == "hist_gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                min_samples_leaf=2,
                random_state=self.seed,
            )

        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=self.seed)

    def fit(self, X_pairs: np.ndarray, y_labels: np.ndarray) -> "SupervisedCandidateRanker":
        """
        Trains the ranker on feature vectors of (Reactants, Candidate) pairs with binary labels (1=true product, 0=decoy).
        """
        self.model.fit(X_pairs, y_labels)
        self.is_trained = True
        return self

    def score_candidates(
        self,
        reactants_smiles: str,
        candidate_smiles_list: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Scores a list of product candidates for a given reaction and returns sorted (candidate, score) pairs.
        """
        if not self.is_trained:
            raise RuntimeError("Candidate ranker is untrained.")
        if not candidate_smiles_list:
            return []

        features = [
            self.featurizer.featurize_candidate_pair(reactants_smiles, cand)
            for cand in candidate_smiles_list
        ]
        X = np.stack(features, axis=0)

        # Get probability of class 1 (true product)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.decision_function(X)

        ranked = sorted(
            zip(candidate_smiles_list, [float(p) for p in probs]),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked

    def evaluate_ranking(
        self,
        eval_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates ranking accuracy across test cases.
        Each case is a dict with:
            'reactants': str,
            'true_product': str,
            'candidates': List[str] (contains true product and decoys)
        """
        top1_hits = 0
        top3_hits = 0
        reciprocal_ranks = []
        latencies = []

        for case in eval_cases:
            r = case["reactants"]
            true_p = canonicalize_smiles(case["true_product"]) or case["true_product"]
            cands = case["candidates"]

            t0 = time.time()
            scored = self.score_candidates(r, cands)
            latencies.append((time.time() - t0) * 1000.0)

            ranked_smiles = [canonicalize_smiles(s) or s for s, _ in scored]
            if true_p in ranked_smiles:
                rank = ranked_smiles.index(true_p) + 1
                reciprocal_ranks.append(1.0 / rank)
                if rank == 1:
                    top1_hits += 1
                if rank <= 3:
                    top3_hits += 1
            else:
                reciprocal_ranks.append(0.0)

        n = max(1, len(eval_cases))
        return {
            "cases_evaluated": n,
            "top1_ranking_accuracy": round(top1_hits / n, 4),
            "top3_ranking_accuracy": round(top3_hits / n, 4),
            "mean_reciprocal_rank_mrr": round(float(np.mean(reciprocal_ranks)), 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 3),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists trained ranker."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "seed": self.seed,
            "fp_nbits": self.fp_nbits,
            "is_trained": self.is_trained,
        }
        joblib.dump(artifact, p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SupervisedCandidateRanker":
        """Loads persisted ranker."""
        artifact = joblib.load(filepath)
        instance = cls(
            model_type=artifact["model_type"],
            seed=artifact["seed"],
            fp_nbits=artifact["fp_nbits"],
        )
        instance.model = artifact["model"]
        instance.is_trained = artifact["is_trained"]
        return instance
