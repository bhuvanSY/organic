"""
Interactive Inference Script for Classical ML Baselines in ChemRxn-Intelligence.
Executes Reaction-Class Prediction, Nearest-Neighbor Retrieval, and Candidate Ranking.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np

from chemrxn.baselines.candidate_ranker import SupervisedCandidateRanker
from chemrxn.baselines.featurizer import ReactionFeaturizer
from chemrxn.baselines.nearest_neighbor import NearestNeighborRetriever
from chemrxn.baselines.reaction_classifier import ReactionClassClassifier
from chemrxn.baselines.taxonomy import ID_TO_CLASS
from chemrxn.chemistry.smiles import canonicalize_smiles, is_valid_smiles, strip_atom_mapping


def run_inference(
    reactant_smiles: str,
    reagents_smiles: str = "",
    candidate_products: Optional[List[str]] = None,
    checkpoint_dir: str = "checkpoints/baselines",
    top_k: int = 5,
):
    print("=" * 70)
    print("  CHEMRXN-INTELLIGENCE: CLASSICAL BASELINE INFERENCE")
    print("=" * 70)

    # Clean and validate input
    clean_r = canonicalize_smiles(strip_atom_mapping(reactant_smiles))
    if not clean_r:
        print(f"ERROR: Invalid reactant SMILES: '{reactant_smiles}'")
        return

    clean_rg = canonicalize_smiles(strip_atom_mapping(reagents_smiles)) if reagents_smiles else ""
    ckpt_dir = Path(checkpoint_dir)

    print(f"Input Reactants : {clean_r}")
    print(f"Input Reagents  : {clean_rg if clean_rg else 'None specified'}")

    featurizer = ReactionFeaturizer(fp_nbits=1024, include_descriptors=True)
    r_feat = featurizer.featurize_reactants(clean_r, clean_rg).reshape(1, -1)

    # 1. Reaction-Class Classifier
    cls_path = ckpt_dir / "reaction_classifier.joblib"
    if cls_path.exists():
        classifier = ReactionClassClassifier.load(cls_path)
        probs = classifier.predict_proba(r_feat)[0]
        top_idx = int(np.argmax(probs))
        top_class = ID_TO_CLASS.get(top_idx, "Unknown")
        top_prob = float(probs[top_idx])

        print("\n--- 1. Reaction-Class Classification (XGBoost/GBDT) ---")
        print(f"Predicted Class : {top_class} (Confidence: {top_prob * 100:.2f}%)")
        print("Top Probability Distribution:")
        sorted_indices = np.argsort(probs)[::-1][:4]
        for idx in sorted_indices:
            print(f"  - {ID_TO_CLASS.get(int(idx), 'Unknown'):35s}: {probs[idx] * 100:5.2f}%")
        print("[NOTE: This predicts the reaction category; it does not generate product SMILES.]")
    else:
        print(f"\n[Classifier checkpoint not found at {cls_path}. Run training first.]")

    # 2. Nearest-Neighbor Reaction Retrieval
    retriever_path = ckpt_dir / "nearest_neighbor.joblib"
    if retriever_path.exists():
        retriever = NearestNeighborRetriever.load(retriever_path)
        retrieved = retriever.retrieve(clean_r, top_k=top_k)

        print(f"\n--- 2. Nearest-Neighbor Historical Reaction Retrieval (Top-{top_k}) ---")
        for item in retrieved:
            print(f"Rank {item['rank']} [Sim: {item['similarity_score']:.4f}]:")
            print(f"  ID        : {item['reaction_id']}")
            print(f"  Reactants : {item['retrieved_reactants']}")
            print(f"  Product   : {item['retrieved_product']}")
    else:
        print(f"\n[Retriever checkpoint not found at {retriever_path}. Run training first.]")

    # 3. Supervised Candidate Ranker
    ranker_path = ckpt_dir / "candidate_ranker.joblib"
    if candidate_products and ranker_path.exists():
        ranker = SupervisedCandidateRanker.load(ranker_path)
        scored = ranker.score_candidates(clean_r, candidate_products)

        print(f"\n--- 3. Supervised Candidate Ranking ---")
        for rank, (cand, score) in enumerate(scored, 1):
            print(f"Rank {rank} [Plausibility: {score:.4f}]: {cand}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run inference using classical ML baselines")
    parser.add_argument("--reactants", type=str, required=True, help="Reactant SMILES string (e.g. 'CC(=O)Cl.CCN')")
    parser.add_argument("--reagents", type=str, default="", help="Reagents SMILES string")
    parser.add_argument("--candidates", type=str, nargs="*", default=None, help="Candidate product SMILES list to rank")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    candidate_list = None
    if args.candidates:
        candidate_list = []
        for c in args.candidates:
            candidate_list.extend([part.strip() for part in c.split(",") if part.strip()])

    run_inference(
        reactant_smiles=args.reactants,
        reagents_smiles=args.reagents,
        candidate_products=candidate_list,
        checkpoint_dir=args.checkpoint_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
