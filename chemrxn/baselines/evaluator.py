"""
Evaluation engine for Classical ML baselines in ChemRxn-Intelligence.
Computes Accuracy, Macro-F1, per-class metrics, retrieval hit rates, and latency.
Generates structured JSON and comprehensive Markdown reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from chemrxn.baselines.taxonomy import ID_TO_CLASS, REACTION_CLASSES


class BaselineEvaluator:
    """
    Computes rigorous empirical evaluation metrics for all classical baselines.
    """

    @staticmethod
    def evaluate_classifier(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[List[str]] = None,
        latency_ms_per_sample: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates classification metrics: Accuracy, Macro-F1, and per-class precision/recall/F1.
        """
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=[ID_TO_CLASS.get(i, f"Class_{i}") for i in sorted(list(set(y_true) | set(y_pred)))],
            output_dict=True,
            zero_division=0,
        )

        per_class = {}
        for k, v in report_dict.items():
            if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg"):
                per_class[k] = {
                    "precision": round(v["precision"], 4),
                    "recall": round(v["recall"], 4),
                    "f1_score": round(v["f1-score"], 4),
                    "support": int(v["support"]),
                }

        return {
            "sample_count": len(y_true),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "latency_ms_per_sample": round(latency_ms_per_sample, 3),
            "per_class_metrics": per_class,
        }

    @staticmethod
    def save_reports(
        summary_data: Dict[str, Any],
        json_path: Path,
        md_path: Path,
    ) -> None:
        """
        Saves structured JSON and presentation Markdown reports to disk.
        """
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Build clean scientific Markdown
        c_res = summary_data.get("reaction_classifier", {})
        r_res = summary_data.get("nearest_neighbor_retrieval", {})
        rk_res = summary_data.get("candidate_ranker", {})

        md = []
        md.append("# ChemRxn-Intelligence: Classical ML Baseline Evaluation Report\n")
        md.append("> **Empirical Baseline Analysis without Pretrained Weights**\n")
        md.append(f"- **Execution Timestamp**: {summary_data.get('timestamp', 'N/A')}")
        md.append(f"- **Seed Determinism**: {summary_data.get('seed', 42)}")
        md.append(f"- **Dataset Source**: {summary_data.get('dataset_source', 'N/A')}\n")

        md.append("## 1. Critical Architectural Distinction: Classification vs. Generation\n")
        md.append("> [!IMPORTANT]")
        md.append("> **Reaction-Class Prediction vs. Product Prediction**:")
        md.append("> - **Reaction Classification** predicts which of the 10 transformation types (e.g., Acylation, Reduction, C-C Coupling) applies to the input reactants. It **never generates or synthesizes product SMILES**.")
        md.append("> - **Nearest-Neighbor Retrieval & Candidate Ranking** serve as the product-level baseline by searching historical patent reactions or re-ranking proposed product molecules.")
        md.append("> - **Generative Product Prediction** (custom Seq2Seq Transformer & Graph Neural Network) will be evaluated on exact-match SMILES sequence generation against these baselines.\n")

        md.append("## 2. Reaction-Class Classifier Results (XGBoost / GBDT)\n")
        md.append(f"- **Test Samples Evaluated**: {c_res.get('sample_count', 0):,}")
        md.append(f"- **Overall Accuracy**: **{c_res.get('accuracy', 0.0) * 100:.2f}%**")
        md.append(f"- **Macro-F1 Score**: **{c_res.get('macro_f1', 0.0):.4f}**")
        md.append(f"- **Weighted-F1 Score**: **{c_res.get('weighted_f1', 0.0):.4f}**")
        md.append(f"- **Inference Latency**: **{c_res.get('latency_ms_per_sample', 0.0)} ms/sample**\n")

        md.append("### Per-Class Performance Breakdown\n")
        md.append("| Reaction Category | Precision | Recall | F1-Score | Support |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for cls_name, metrics in c_res.get("per_class_metrics", {}).items():
            md.append(f"| {cls_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['support']} |")
        md.append("")

        md.append("## 3. Nearest-Neighbor Reaction Retrieval Baseline\n")
        md.append(f"- **Database Historical Reactions**: {r_res.get('database_size', 0):,}")
        md.append(f"- **Test Query Samples**: {r_res.get('sample_count', 0):,}")
        md.append(f"- **Top-1 Exact Product Hit Rate**: **{r_res.get('top1_hit_rate', 0.0) * 100:.2f}%**")
        md.append(f"- **Top-3 Exact Product Hit Rate**: **{r_res.get('top3_hit_rate', 0.0) * 100:.2f}%**")
        md.append(f"- **Top-5 Exact Product Hit Rate**: **{r_res.get('top5_hit_rate', 0.0) * 100:.2f}%**")
        md.append(f"- **Mean Top-1 Tanimoto Similarity**: **{r_res.get('mean_top1_tanimoto_similarity', 0.0):.4f}**")
        md.append(f"- **Retrieval Latency**: **{r_res.get('latency_ms_per_query', 0.0)} ms/query**\n")

        md.append("## 4. Supervised Candidate-Ranking Baseline\n")
        md.append(f"- **Cases Evaluated**: {rk_res.get('cases_evaluated', 0):,}")
        md.append(f"- **Top-1 Candidate Selection Accuracy**: **{rk_res.get('top1_ranking_accuracy', 0.0) * 100:.2f}%**")
        md.append(f"- **Top-3 Candidate Selection Accuracy**: **{rk_res.get('top3_ranking_accuracy', 0.0) * 100:.2f}%**")
        md.append(f"- **Mean Reciprocal Rank (MRR)**: **{rk_res.get('mean_reciprocal_rank_mrr', 0.0):.4f}**")
        md.append(f"- **Ranking Latency**: **{rk_res.get('mean_latency_ms', 0.0)} ms/case**\n")

        md.append("## 5. Comparative Framework: Baseline vs. Custom Neural Architectures\n")
        md.append("| Model Family | Paradigm | Input Representation | Output Target | Strengths | Limitations |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        md.append("| **Reaction Classifier** | Classical ML (XGBoost) | Morgan FP (1024) + Descriptors | Categorical Label (10 classes) | High throughput, interpretable | Cannot generate novel molecules |")
        md.append("| **Nearest-Neighbor** | Instance Retrieval | Cosine on ECFP4 vectors | Retrieved Historical Product | Non-parametric, zero training | Poor generalization on novel scaffolds |")
        md.append("| **Candidate Ranker** | Discriminative GBDT | Pair Difference Fingerprint | Plausibility Probability | Filters spurious proposals | Relies on external candidate generator |")
        md.append("| **Custom Transformer** | End-to-End Generative | Tokenized Reactant SMILES | Auto-regressive Product SMILES | Novel molecule generation, atom-rearrangement | Requires significant training epochs |")
        md.append("| **Graph Neural Model** | Geometric Message Passing | Molecular Atom-Bond Graph | Atom-center reactivity edits | Chemically bounded, physically rigorous | Graph parsing overhead |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
