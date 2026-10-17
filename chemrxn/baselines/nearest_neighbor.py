"""
Nearest-Neighbor Reaction Retrieval Baseline using Molecular Fingerprints.
Retrieves historically analogous reactions and outputs retrieved products as synthesis candidates.
Evaluates empirical Top-k product hit rate and query latency.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from chemrxn.baselines.featurizer import ReactionFeaturizer
from chemrxn.chemistry.fingerprints import compute_tanimoto_similarity
from chemrxn.chemistry.smiles import canonicalize_smiles


class NearestNeighborRetriever:
    """
    Instance-based non-parametric retrieval baseline for synthesis prediction.
    Given query reactants, retrieves nearest reactions and uses historical products as candidates.
    """

    def __init__(
        self,
        metric: str = "cosine",
        fp_nbits: int = 1024,
        fp_radius: int = 2,
    ):
        self.metric = metric
        self.fp_nbits = fp_nbits
        self.fp_radius = fp_radius
        self.featurizer = ReactionFeaturizer(fp_radius=fp_radius, fp_nbits=fp_nbits, include_descriptors=False)
        self.index: Optional[NearestNeighbors] = None
        self.database_reactants: List[str] = []
        self.database_products: List[str] = []
        self.database_ids: List[str] = []
        self.database_fps: Optional[np.ndarray] = None

    def fit(
        self,
        reactants_list: List[str],
        products_list: List[str],
        reaction_ids: Optional[List[str]] = None,
        fps_matrix: Optional[np.ndarray] = None,
    ) -> "NearestNeighborRetriever":
        """Builds retrieval index from historical reaction database."""
        self.database_reactants = list(reactants_list)
        self.database_products = list(products_list)
        self.database_ids = list(reaction_ids) if reaction_ids else [f"db_{i}" for i in range(len(reactants_list))]

        if fps_matrix is not None:
            self.database_fps = np.asarray(fps_matrix, dtype=np.float32)
        else:
            fps = [self.featurizer.mol_fingerprint(r) for r in self.database_reactants]
            self.database_fps = np.stack(fps, axis=0).astype(np.float32)

        norms = np.linalg.norm(self.database_fps, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.database_fps_norm = (self.database_fps / norms).astype(np.float32)
        return self

    def retrieve(self, query_reactants: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k closest historical reactions for a query reactant string using cosine similarity.
        """
        if self.database_fps_norm is None or len(self.database_reactants) == 0:
            raise RuntimeError("Retriever index has not been built.")

        q_fp = self.featurizer.mol_fingerprint(query_reactants).astype(np.float32)
        q_norm = np.linalg.norm(q_fp)
        if q_norm > 0:
            q_fp = q_fp / q_norm

        sims = np.dot(self.database_fps_norm, q_fp)
        k = min(top_k, len(self.database_reactants))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = []
        for idx in top_idx:
            results.append({
                "rank": len(results) + 1,
                "reaction_id": self.database_ids[idx],
                "retrieved_reactants": self.database_reactants[idx],
                "retrieved_product": self.database_products[idx],
                "similarity_score": round(float(sims[idx]), 4),
            })
        return results

    def batch_evaluate(
        self,
        test_reactants: List[str],
        test_true_products: List[str],
        top_k_list: Tuple[int, ...] = (1, 3, 5),
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval hit rate: does the top-k retrieved products contain the true product?
        Uses GPU-accelerated batch matrix multiplication when available.
        """
        if not test_reactants or not test_true_products:
            return {}

        max_k = max(top_k_list)
        total = len(test_reactants)
        hits = {k: 0 for k in top_k_list}
        tanimoto_scores: List[float] = []

        start_time = time.time()
        test_canon_products = [canonicalize_smiles(p) or p for p in test_true_products]

        # Featurize test queries
        q_fps = np.stack([self.featurizer.mol_fingerprint(r) for r in test_reactants], axis=0).astype(np.float32)
        q_norms = np.linalg.norm(q_fps, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        q_fps_norm = q_fps / q_norms

        # Check for GPU acceleration
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available() and self.database_fps_norm is not None:
                use_gpu = True
                db_t = torch.tensor(self.database_fps_norm, device="cuda", dtype=torch.float32)
                q_t = torch.tensor(q_fps_norm, device="cuda", dtype=torch.float32)

                # Process in blocks of 2000 queries to keep VRAM minimal
                block_size = 2000
                all_top_indices = []
                for b_start in range(0, total, block_size):
                    q_block = q_t[b_start : b_start + block_size]
                    sim_mat = torch.mm(q_block, db_t.T)
                    _, b_indices = torch.topk(sim_mat, max_k, dim=1)
                    all_top_indices.append(b_indices.cpu().numpy())
                top_indices_matrix = np.concatenate(all_top_indices, axis=0)
        except Exception:
            use_gpu = False

        if not use_gpu:
            # Fallback to chunked numpy dot product
            all_top_indices = []
            block_size = 500
            for b_start in range(0, total, block_size):
                q_block = q_fps_norm[b_start : b_start + block_size]
                sim_mat = np.dot(q_block, self.database_fps_norm.T)
                b_indices = np.argpartition(sim_mat, -max_k, axis=1)[:, -max_k:]
                sorted_indices = []
                for row_idx, row in enumerate(sim_mat):
                    best_k = b_indices[row_idx]
                    best_k_sorted = best_k[np.argsort(-row[best_k])]
                    sorted_indices.append(best_k_sorted)
                all_top_indices.append(np.array(sorted_indices))
            top_indices_matrix = np.concatenate(all_top_indices, axis=0)

        # Calculate hit rates & Tanimoto similarities
        for i in range(total):
            true_p_canon = test_canon_products[i]
            retrieved_indices = top_indices_matrix[i]
            retrieved_prods = [self.database_products[idx] for idx in retrieved_indices]

            for k in top_k_list:
                candidates_k = retrieved_prods[:k]
                if any(canonicalize_smiles(c) == true_p_canon for c in candidates_k):
                    hits[k] += 1

            top1_p = retrieved_prods[0] if retrieved_prods else ""
            tanimoto_scores.append(compute_tanimoto_similarity(top1_p, true_p_canon))

        elapsed_time = time.time() - start_time
        latency_ms_per_query = round((elapsed_time / total) * 1000.0, 3)

        return {
            "sample_count": total,
            "latency_ms_per_query": latency_ms_per_query,
            "total_eval_seconds": round(elapsed_time, 2),
            "top1_hit_rate": round(hits.get(1, 0) / total, 4),
            "top3_hit_rate": round(hits.get(3, 0) / total, 4),
            "top5_hit_rate": round(hits.get(5, 0) / total, 4),
            "mean_top1_tanimoto_similarity": round(float(np.mean(tanimoto_scores)), 4),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists database and index."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "metric": self.metric,
            "fp_nbits": self.fp_nbits,
            "fp_radius": self.fp_radius,
            "database_reactants": self.database_reactants,
            "database_products": self.database_products,
            "database_ids": self.database_ids,
            "database_fps_norm": self.database_fps_norm,
        }
        joblib.dump(artifact, p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NearestNeighborRetriever":
        """Loads persisted retriever."""
        artifact = joblib.load(filepath)
        instance = cls(
            metric=artifact["metric"],
            fp_nbits=artifact["fp_nbits"],
            fp_radius=artifact["fp_radius"],
        )
        instance.database_reactants = artifact["database_reactants"]
        instance.database_products = artifact["database_products"]
        instance.database_ids = artifact["database_ids"]
        instance.database_fps_norm = artifact.get("database_fps_norm", None)
        if instance.database_fps_norm is None and "database_fps" in artifact:
            fps = artifact["database_fps"]
            norms = np.linalg.norm(fps, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            instance.database_fps_norm = (fps / norms).astype(np.float32)
        return instance
