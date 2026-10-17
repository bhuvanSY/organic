"""
Baseline Pipeline Runner for ChemRxn-Intelligence.
Trains, evaluates, and persists:
1. Reaction-Class Classifier (XGBoost / GBDT)
2. Nearest-Neighbor Reaction Retrieval (Morgan FP)
3. Supervised Candidate Ranker
Generates comprehensive JSON and Markdown reports.
"""

import argparse
import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from chemrxn.baselines.candidate_ranker import SupervisedCandidateRanker
from chemrxn.baselines.evaluator import BaselineEvaluator
from chemrxn.baselines.featurizer import ReactionFeaturizer
from chemrxn.baselines.nearest_neighbor import NearestNeighborRetriever
from chemrxn.baselines.reaction_classifier import ReactionClassClassifier
from chemrxn.baselines.taxonomy import CLASS_TO_ID, ReactionClassifierTagger
from chemrxn.utils.logger import get_logger, setup_logging
from chemrxn.utils.seed import seed_everything

logger = get_logger("baselines")


class ClassicalBaselineRunner:
    """
    Orchestrates the classical ML baselines training, validation, evaluation, and reporting.
    """

    def __init__(
        self,
        train_path: str = "data/processed/train.parquet",
        test_path: str = "data/processed/test.parquet",
        output_dir: str = "reports",
        checkpoint_dir: str = "checkpoints/baselines",
        classifier_type: str = "xgboost",
        train_limit: Optional[int] = 5000,
        test_limit: Optional[int] = 1000,
        seed: int = 42,
    ):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.classifier_type = classifier_type
        self.train_limit = train_limit
        self.test_limit = test_limit
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tagger = ReactionClassifierTagger()
        self.featurizer = ReactionFeaturizer(fp_nbits=1024, include_descriptors=True)

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and samples train and test parquet partitions."""
        if not self.train_path.exists() or not self.test_path.exists():
            raise FileNotFoundError(f"Processed partitions not found at {self.train_path} or {self.test_path}")

        logger.info(f"Loading datasets from {self.train_path} and {self.test_path}...")
        df_train = pd.read_parquet(self.train_path)
        df_test = pd.read_parquet(self.test_path)

        if self.train_limit is not None and self.train_limit > 0 and len(df_train) > self.train_limit:
            df_train = df_train.sample(n=self.train_limit, random_state=self.seed).reset_index(drop=True)
        if self.test_limit is not None and self.test_limit > 0 and len(df_test) > self.test_limit:
            df_test = df_test.sample(n=self.test_limit, random_state=self.seed).reset_index(drop=True)

        logger.info(f"Active dataset: {len(df_train):,} train samples, {len(df_test):,} test samples.")
        return df_train, df_test

    def run(self) -> Dict[str, Any]:
        """Executes full classical ML baseline suite."""
        seed_everything(self.seed)
        start_time = time.time()
        logger.info("Starting Classical ML Baseline Pipeline...")

        df_train, df_test = self.load_datasets()

        # -------------------------------------------------------------
        # 1. Feature Extraction & Reaction Class Assignment
        # -------------------------------------------------------------
        logger.info(f"Extracting reaction features for {len(df_train):,} train samples...")
        batch_size = 50000
        train_classes: List[str] = []
        train_feats: List[np.ndarray] = []

        for b_start in range(0, len(df_train), batch_size):
            b_df = df_train.iloc[b_start : b_start + batch_size]
            t0 = time.time()
            b_classes = [
                self.tagger.classify_reaction(r, p, rg)
                for r, p, rg in zip(b_df["reactants_smiles"], b_df["products_smiles"], b_df["reagents_smiles"])
            ]
            b_features = [
                self.featurizer.featurize_reactants(r, rg)
                for r, rg in zip(b_df["reactants_smiles"], b_df["reagents_smiles"])
            ]
            train_classes.extend(b_classes)
            train_feats.append(np.stack(b_features, axis=0).astype(np.float32))
            logger.info(f"Train featurization: {min(b_start + batch_size, len(df_train)):,}/{len(df_train):,} completed ({time.time() - t0:.1f}s)")

        X_train = np.concatenate(train_feats, axis=0)
        y_train = np.array([CLASS_TO_ID.get(c, 0) for c in train_classes])

        logger.info(f"Extracting reaction features for {len(df_test):,} test samples...")
        test_classes = [
            self.tagger.classify_reaction(r, p, rg)
            for r, p, rg in zip(df_test["reactants_smiles"], df_test["products_smiles"], df_test["reagents_smiles"])
        ]
        test_features = [
            self.featurizer.featurize_reactants(r, rg)
            for r, rg in zip(df_test["reactants_smiles"], df_test["reagents_smiles"])
        ]
        X_test = np.stack(test_features, axis=0).astype(np.float32)
        y_test = np.array([CLASS_TO_ID.get(c, 0) for c in test_classes])

        # -------------------------------------------------------------
        # 2. Train & Evaluate Reaction-Class Classifier
        # -------------------------------------------------------------
        logger.info(f"Training Reaction-Class Classifier ({self.classifier_type.upper()}) on {len(X_train):,} samples...")
        classifier = ReactionClassClassifier(model_type=self.classifier_type, seed=self.seed)
        classifier.fit(X_train, y_train)

        t0_inf = time.time()
        y_pred = classifier.predict(X_test)
        latency_cls = ((time.time() - t0_inf) / len(X_test)) * 1000.0

        cls_report = BaselineEvaluator.evaluate_classifier(
            y_true=y_test,
            y_pred=y_pred,
            latency_ms_per_sample=latency_cls,
        )
        classifier_path = self.checkpoint_dir / "reaction_classifier.joblib"
        classifier.save(classifier_path)
        logger.info(f"Classifier Accuracy: {cls_report['accuracy'] * 100:.2f}%, Macro-F1: {cls_report['macro_f1']:.4f}")

        # -------------------------------------------------------------
        # 3. Fit & Evaluate Nearest-Neighbor Retrieval Baseline
        # -------------------------------------------------------------
        logger.info(f"Indexing {len(df_train):,} historical reactions for Nearest-Neighbor retrieval...")
        retriever = NearestNeighborRetriever(metric="cosine", fp_nbits=1024)
        retriever.fit(
            reactants_list=df_train["reactants_smiles"].tolist(),
            products_list=df_train["products_smiles"].tolist(),
            reaction_ids=df_train["reaction_id"].tolist(),
            fps_matrix=X_train[:, :1024],
        )

        logger.info(f"Evaluating retrieval hit rates on {len(df_test):,} test queries...")
        retrieval_report = retriever.batch_evaluate(
            test_reactants=df_test["reactants_smiles"].tolist(),
            test_true_products=df_test["products_smiles"].tolist(),
            top_k_list=(1, 3, 5),
        )
        retrieval_report["database_size"] = len(df_train)
        retriever_path = self.checkpoint_dir / "nearest_neighbor.joblib"
        retriever.save(retriever_path)
        logger.info(f"Retrieval Top-1 Hit Rate: {retrieval_report['top1_hit_rate'] * 100:.2f}%, Mean Tanimoto: {retrieval_report['mean_top1_tanimoto_similarity']:.4f}")

        # -------------------------------------------------------------
        # 4. Supervised Candidate Ranker Baseline
        # -------------------------------------------------------------
        logger.info("Preparing candidate ranking evaluation...")
        ranker_train_pairs = []
        ranker_train_labels = []

        train_pool_products = df_train["products_smiles"].tolist()
        rng = np.random.RandomState(self.seed)

        n_rank_train = min(5000, len(df_train))
        for r_smi, true_p in zip(df_train["reactants_smiles"][:n_rank_train], df_train["products_smiles"][:n_rank_train]):
            ranker_train_pairs.append(self.featurizer.featurize_candidate_pair(r_smi, true_p))
            ranker_train_labels.append(1)

            decoys = rng.choice(train_pool_products, size=3, replace=False)
            for d in decoys:
                if d != true_p:
                    ranker_train_pairs.append(self.featurizer.featurize_candidate_pair(r_smi, d))
                    ranker_train_labels.append(0)

        X_rank_train = np.stack(ranker_train_pairs, axis=0).astype(np.float32)
        y_rank_train = np.array(ranker_train_labels)

        ranker = SupervisedCandidateRanker(model_type=self.classifier_type, seed=self.seed)
        ranker.fit(X_rank_train, y_rank_train)

        # Build test ranking cases
        test_cases = []
        n_rank_test = min(500, len(df_test))
        for r_smi, true_p in zip(df_test["reactants_smiles"][:n_rank_test], df_test["products_smiles"][:n_rank_test]):
            retrieved = retriever.retrieve(r_smi, top_k=4)
            cands = [item["retrieved_product"] for item in retrieved]
            if true_p not in cands:
                cands.append(true_p)
            rng.shuffle(cands)
            test_cases.append({
                "reactants": r_smi,
                "true_product": true_p,
                "candidates": cands,
            })

        ranking_report = ranker.evaluate_ranking(test_cases)
        ranker_path = self.checkpoint_dir / "candidate_ranker.joblib"
        ranker.save(ranker_path)
        logger.info(f"Candidate Ranker Top-1 Accuracy: {ranking_report['top1_ranking_accuracy'] * 100:.2f}%, MRR: {ranking_report['mean_reciprocal_rank_mrr']:.4f}")

        # -------------------------------------------------------------
        # 5. Compile Baseline Report & Save Artifacts
        # -------------------------------------------------------------
        total_time = round(time.time() - start_time, 2)
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": self.seed,
            "dataset_source": str(self.train_path),
            "total_execution_seconds": total_time,
            "train_samples_used": len(df_train),
            "test_samples_used": len(df_test),
            "reaction_classifier": cls_report,
            "nearest_neighbor_retrieval": retrieval_report,
            "candidate_ranker": ranking_report,
            "saved_checkpoints": {
                "classifier": str(classifier_path),
                "retriever": str(retriever_path),
                "candidate_ranker": str(ranker_path),
            },
        }

        json_out = self.output_dir / "classical_baseline_report.json"
        md_out = self.output_dir / "classical_baseline_report.md"
        BaselineEvaluator.save_reports(summary, json_out, md_out)

        logger.info(f"Baseline reports saved to {json_out} and {md_out} in {total_time}s.")
        return summary


def main():
    parser = argparse.ArgumentParser(description="ChemRxn-Intelligence Classical ML Baselines")
    parser.add_argument("--train-path", type=str, default="data/processed/train.parquet")
    parser.add_argument("--test-path", type=str, default="data/processed/test.parquet")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--classifier-type", type=str, choices=["xgboost", "hist_gb", "random_forest"], default="xgboost")
    parser.add_argument("--train-samples", type=int, default=0, help="Number of train samples (0 for full dataset)")
    parser.add_argument("--test-samples", type=int, default=2000, help="Number of test samples (0 for full test set)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging(level="INFO")

    runner = ClassicalBaselineRunner(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        classifier_type=args.classifier_type,
        train_limit=args.train_samples,
        test_limit=args.test_samples,
        seed=args.seed,
    )

    results = runner.run()
    c = results["reaction_classifier"]
    r = results["nearest_neighbor_retrieval"]
    rk = results["candidate_ranker"]

    print("\n" + "=" * 70)
    print("  CHEMRXN-INTELLIGENCE: CLASSICAL ML BASELINES SUMMARY")
    print("=" * 70)
    print(f"1. Reaction-Class Classifier (XGBoost / GBDT):")
    print(f"   - Overall Accuracy    : {c['accuracy'] * 100:.2f}%")
    print(f"   - Macro-F1 Score      : {c['macro_f1']:.4f}")
    print(f"   - Weighted-F1 Score   : {c['weighted_f1']:.4f}")
    print(f"   - Inference Latency   : {c['latency_ms_per_sample']} ms/sample")
    print(f"\n2. Nearest-Neighbor Reaction Retrieval:")
    print(f"   - Top-1 Product Hit   : {r['top1_hit_rate'] * 100:.2f}%")
    print(f"   - Top-3 Product Hit   : {r['top3_hit_rate'] * 100:.2f}%")
    print(f"   - Top-5 Product Hit   : {r['top5_hit_rate'] * 100:.2f}%")
    print(f"   - Mean Top-1 Tanimoto : {r['mean_top1_tanimoto_similarity']:.4f}")
    print(f"   - Retrieval Latency   : {r['latency_ms_per_query']} ms/query")
    print(f"\n3. Supervised Candidate Ranker:")
    print(f"   - Top-1 Accuracy      : {rk['top1_ranking_accuracy'] * 100:.2f}%")
    print(f"   - Top-3 Accuracy      : {rk['top3_ranking_accuracy'] * 100:.2f}%")
    print(f"   - MRR                 : {rk['mean_reciprocal_rank_mrr']:.4f}")
    print(f"   - Latency             : {rk['mean_latency_ms']} ms/case")
    print(f"\nReports Generated:")
    print(f"   - JSON    : {Path(args.output_dir) / 'classical_baseline_report.json'}")
    print(f"   - Markdown: {Path(args.output_dir) / 'classical_baseline_report.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
