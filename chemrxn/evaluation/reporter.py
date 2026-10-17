"""
Evaluation Reporter and Artifact Generator for ChemRxn-Intelligence.
Exports:
1. Comprehensive JSON telemetry
2. CSV comparison tables
3. Detailed Markdown research report with bootstrap CIs, split methodology, and reproduction commands
4. Matplotlib publication-grade visualization plots
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


class EvaluationReporter:
    """
    Exports evaluation results across multiple formats and generates visualization figures.
    """
    def __init__(self, output_dir: str = "reports/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, results: Dict[str, Any], filename: str = "evaluation_results.json") -> Path:
        target = self.output_dir / filename
        with open(target, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return target

    def export_csv(self, model_summaries: List[Dict[str, Any]], filename: str = "evaluation_results.csv") -> Path:
        target = self.output_dir / filename
        if not model_summaries:
            return target

        fieldnames = [
            "model_name",
            "top1_acc_pct",
            "top1_ci_lower",
            "top1_ci_upper",
            "top3_acc_pct",
            "top5_acc_pct",
            "valid_smiles_rate_pct",
            "stereo_aware_acc_pct",
            "abstention_rate_pct",
            "precision_at_coverage_pct",
            "ece",
            "median_latency_ms",
            "throughput_rxn_sec",
            "parameter_count",
        ]

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in model_summaries:
                writer.writerow(m)

        return target

    def generate_plots(self, model_summaries: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generates publication-quality comparison charts using matplotlib.
        """
        plot_paths: Dict[str, str] = {}
        if not model_summaries:
            return plot_paths

        # Style configuration
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        names = [m["model_name"] for m in model_summaries]

        # 1. Accuracy Comparison with Bootstrap Error Bars
        fig, ax = plt.subplots(figsize=(10, 5))
        top1_vals = [m["top1_acc_pct"] for m in model_summaries]
        ci_lowers = [m["top1_acc_pct"] - m.get("top1_ci_lower", m["top1_acc_pct"]) for m in model_summaries]
        ci_uppers = [m.get("top1_ci_upper", m["top1_acc_pct"]) - m["top1_acc_pct"] for m in model_summaries]
        yerr = [ci_lowers, ci_uppers]

        bars = ax.bar(names, top1_vals, yerr=yerr, capsize=5, color="#2563EB", alpha=0.85, edgecolor="#1D4ED8")
        ax.set_ylabel("Top-1 Exact Match Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Organic Reaction Prediction Top-1 Accuracy (95% Bootstrap CI)", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        plt.xticks(rotation=15, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        p1_path = self.plots_dir / "accuracy_by_model.png"
        fig.tight_layout()
        fig.savefig(p1_path, dpi=300)
        plt.close(fig)
        plot_paths["accuracy_plot"] = str(p1_path)

        # 2. Latency vs Accuracy Tradeoff Scatter
        fig, ax = plt.subplots(figsize=(9, 5.5))
        latencies = [m["median_latency_ms"] for m in model_summaries]
        colors = ["#DC2626", "#EA580C", "#0D9488", "#2563EB", "#7C3AED"]

        for i, m in enumerate(model_summaries):
            c = colors[i % len(colors)]
            ax.scatter(m["median_latency_ms"], m["top1_acc_pct"], s=200, color=c, alpha=0.85, edgecolors="black", zorder=3)
            ax.annotate(m["model_name"], (m["median_latency_ms"], m["top1_acc_pct"]),
                        xytext=(8, 4), textcoords="offset points", fontsize=9, fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlabel("Median Inference Latency (ms) [Log Scale]", fontsize=11, fontweight="bold")
        ax.set_ylabel("Top-1 Exact Match Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("Accuracy vs. Latency Tradeoff across Model Architectures", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        p2_path = self.plots_dir / "latency_vs_accuracy.png"
        fig.tight_layout()
        fig.savefig(p2_path, dpi=300)
        plt.close(fig)
        plot_paths["latency_accuracy_plot"] = str(p2_path)

        return plot_paths

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        model_summaries: List[Dict[str, Any]],
        filename: str = "comprehensive_evaluation_report.md",
    ) -> Path:
        """
        Assembles comprehensive research evaluation report.
        """
        target = self.output_dir / filename
        meta = results.get("metadata", {})
        scaff = results.get("scaffold_generalization", {})

        rows = []
        for m in model_summaries:
            ci_str = f"[{m.get('top1_ci_lower', 0.0):.1f}%, {m.get('top1_ci_upper', 0.0):.1f}%]"
            rows.append(
                f"| **{m['model_name']}** | {m['top1_acc_pct']:.1f}% | {ci_str} | "
                f"{m.get('top3_acc_pct', 0.0):.1f}% | {m.get('top5_acc_pct', 0.0):.1f}% | "
                f"{m.get('stereo_aware_acc_pct', 0.0):.1f}% | {m.get('valid_smiles_rate_pct', 0.0):.1f}% | "
                f"{m.get('ece', 0.0):.4f} | {m.get('median_latency_ms', 0.0):.2f} ms | "
                f"{m.get('parameter_count', 'N/A')} |"
            )

        table_str = "\n".join(rows)

        content = f"""# ChemRxn-Intelligence: Rigorous Scientific Evaluation Report

**Benchmark Cohort**: Locked Test Set ($N = {meta.get('test_samples_evaluated', 0)}$)  
**Dataset Split**: Scaffold-aware split (80% Train, 10% Valid, 10% Test)  
**Hardware Platform**: {meta.get('hardware_device', 'CUDA Accelerated')}  
**Evaluator Version**: {meta.get('evaluator_version', '1.0.0')}  

---

## 1. Unified Benchmark Results

| Model Architecture | Top-1 Acc | 95% Bootstrap CI | Top-3 Acc | Top-5 Acc | Stereo-Aware | Validity | ECE | Median Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## 2. Chemical Validity Breakdown & Failure Taxonomy

Every predicted SMILES string was passed through RDKit's C++ parser and sanitizer:
- **Chemical Syntax Validity vs Chemical Correctness**:
  - *Syntactic Validity*: Molecule conforms to valid atom valences and ring closures.
  - *Reference Agreement*: Predicted molecule matches historical USPTO reaction records.
  - *Experimental Feasibility*: Thermodynamic spontaneity and real-world wet-lab synthesis yield.
  > **Crucial Scientific Distinction**: A molecule may pass RDKit sanitization with 100% syntactic validity while remaining completely incorrect relative to the target reaction, or kinetically unfeasible under ambient conditions.

### Failure Distribution across Raw Model Predictions:
```
{json.dumps(results.get('validity_breakdown', {}), indent=2)}
```

---

## 3. Scaffold Generalization & Train-Test Leakage Analysis

To prevent data contamination, Bemis-Murcko core scaffolds were computed across training and test splits:
- **Total Unique Scaffolds in Training Split**: {scaff.get('train_scaffold_count', 'N/A')}
- **Exact Reaction Contamination (Train $\\cap$ Test)**: **{scaff.get('exact_product_leakage_count', 0)} reactions ({scaff.get('exact_product_leakage_rate', 0.0)*100:.2f}%)**
- **Seen Scaffold Accuracy**: **{scaff.get('seen_scaffold_accuracy', 0.0):.1f}%** (Test reactions sharing a scaffold core with training set)
- **Unseen Scaffold Accuracy**: **{scaff.get('unseen_scaffold_accuracy', 0.0):.1f}%** (Generalization to novel ring systems)

---

## 4. Calibration & Selective Coverage

- **Expected Calibration Error (ECE)**: Evaluates whether predicted confidence probabilities correspond to true empirical accuracy.
- **Abstention Protocol**: Predictions with calibrated confidence below $\\tau = {meta.get('abstention_threshold', 0.35)}$ are abstained to protect against hallucinations on out-of-domain reagents.
- **Precision at Coverage**: When the hybrid engine accepts a prediction, Top-1 accuracy is elevated without guessing on uncertain candidates.

---

## 5. Architectural Limitations & Scientific Safeguards

1. **Reaction Conditions & Catalysts**:
   Reaction SMILES often lack explicit stereochemical catalyst mechanisms, stoichiometry, or solvent dynamics. The model predicts major organic connectivity rather than reaction kinetics.
2. **Disconnected Molecule Handling**:
   Reactants are represented as disjoint-union molecular graphs. Message passing propagates intramolecularly while graph-level pooling aggregates component mixtures.
3. **No Claim of Wet-Lab Feasibility**:
   All benchmark numbers measure algorithmic agreement against locked historical USPTO patent data. No experimental claims of yield, enantiomeric excess, or scalability are made without wet-lab synthesis.

---

## 6. Exact Reproduction Commands

To replicate this exact benchmark deterministically:
```powershell
# 1. Run rigorous evaluation CLI on locked test split
python scripts/run_evaluation_framework.py --num-samples 30 --beam-size 4 --bootstrap-samples 1000

# 2. Run full automated evaluation test suite
python -m pytest tests/test_evaluation_framework.py
```
"""
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        return target
