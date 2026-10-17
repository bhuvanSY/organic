"""
ChemRxn-Intelligence: Rigorous Unified Evaluation Framework CLI Driver.

Evaluates all five model families on an identical locked test cohort:
1. Classical ML Baseline (Supervised Ranker / Morgan Fingerprints)
2. Scratch Seq2Seq Transformer (Beam Search)
3. Molecular Graph Neural Network (MPNN Reaction-Pair Scorer)
4. Full Hybrid Inference System (Multi-Modal Fusion + Calibration + Abstention)
5. Experimental RL Policy (Actor-Critic Candidate Selection)

Exports JSON telemetry, CSV summaries, publication-grade figures, and Markdown report.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

from chemrxn.evaluation.benchmark_runner import UnifiedBenchmarkSuite, load_locked_test_dataset
from chemrxn.evaluation.reporter import EvaluationReporter


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Rigorous Evaluation Framework for ChemRxn-Intelligence")
    parser.add_argument("--config", type=str, default="configs/evaluation_config.json", help="Path to evaluation config JSON")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of locked test samples to evaluate")
    parser.add_argument("--beam-size", type=int, default=None, help="Beam size override")
    parser.add_argument("--bootstrap-samples", type=int, default=None, help="Bootstrap samples override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Resolve settings
    beam_size = args.beam_size or config.get("decoding_budget", {}).get("beam_size", 4)
    n_bootstrap = args.bootstrap_samples or config.get("evaluation_parameters", {}).get("n_bootstrap", 1000)
    output_dir = args.output_dir or config.get("outputs", {}).get("report_dir", "reports/evaluation")

    test_path = config.get("dataset", {}).get("test_path", "data/processed/test.parquet")
    if not os.path.exists(test_path):
        test_path = config.get("dataset", {}).get("fallback_test_path", "data/processed/test.csv")

    train_path = config.get("dataset", {}).get("train_path", "data/processed/train.parquet")
    if not os.path.exists(train_path):
        train_path = config.get("dataset", {}).get("fallback_train_path", "data/processed/train.csv")

    vocab_path = config.get("dataset", {}).get("vocab_path", "data/processed/smiles_vocab.json")

    ckpts = config.get("checkpoints", {})
    transformer_ckpt = ckpts.get("transformer", "checkpoints/transformer/best_checkpoint.pt")
    gnn_ckpt = ckpts.get("gnn", "checkpoints/gnn/best_gnn.pt")
    baseline_ckpt = ckpts.get("baseline", "checkpoints/baselines/candidate_ranker.joblib")
    rl_ckpt = ckpts.get("rl_policy", "checkpoints/rl/best_policy.pt")

    print("=" * 80)
    print("ChemRxn-Intelligence: Rigorous Benchmark Evaluation")
    print(f"Test Path:         {test_path}")
    print(f"Train Path:        {train_path}")
    print(f"Test Cohort Size:  {args.num_samples}")
    print(f"Beam Size:         {beam_size}")
    print(f"Bootstrap Resamp:  {n_bootstrap}")
    print(f"Output Directory:  {output_dir}")
    print("=" * 80)

    # 1. Load locked test data
    test_pairs, train_products = load_locked_test_dataset(
        test_path=test_path,
        train_path=train_path,
        max_samples=args.num_samples,
    )

    if not test_pairs:
        print("[ERROR] No valid test reaction pairs loaded. Aborting benchmark.")
        sys.exit(1)

    print(f"[Data Loader] Loaded {len(test_pairs)} locked test reactions and {len(train_products)} training reference products.")

    # 2. Initialize Unified Benchmark Suite
    suite = UnifiedBenchmarkSuite(
        transformer_ckpt=transformer_ckpt,
        gnn_ckpt=gnn_ckpt,
        baseline_ckpt=baseline_ckpt,
        rl_ckpt=rl_ckpt,
        vocab_path=vocab_path,
        beam_size=beam_size,
        n_bootstrap=n_bootstrap,
    )

    # 3. Execute rigorous benchmark
    results = suite.run_benchmark(test_pairs, train_products)

    # 4. Generate Reports & Export Artifacts
    reporter = EvaluationReporter(output_dir=output_dir)
    json_path = reporter.export_json(results, config.get("outputs", {}).get("json_telemetry", "evaluation_results.json"))
    csv_path = reporter.export_csv(results["model_summaries"], config.get("outputs", {}).get("csv_summary", "evaluation_results.csv"))
    plots = reporter.generate_plots(results["model_summaries"])
    md_path = reporter.generate_markdown_report(
        results,
        results["model_summaries"],
        config.get("outputs", {}).get("markdown_report", "comprehensive_evaluation_report.md")
    )

    print("=" * 80)
    print("BENCHMARK EXECUTION COMPLETE")
    print(f"JSON Telemetry:    {json_path}")
    print(f"CSV Summary:       {csv_path}")
    print(f"Markdown Report:   {md_path}")
    for p_name, p_file in plots.items():
        print(f"Generated Plot:    {p_name} -> {p_file}")
    print("=" * 80)

    # Print summary table
    print("\nSummary Results:")
    print(f"{'Model Architecture':<30} | {'Top-1':<7} | {'95% CI':<16} | {'Top-3':<7} | {'Top-5':<7} | {'Valid%':<7} | {'ECE':<7} | {'Median ms':<10}")
    print("-" * 105)
    for m in results["model_summaries"]:
        ci_str = f"[{m.get('top1_ci_lower', 0.0):.1f}%, {m.get('top1_ci_upper', 0.0):.1f}%]"
        print(f"{m['model_name']:<30} | {m['top1_acc_pct']:>6.1f}% | {ci_str:<16} | {m.get('top3_acc_pct', 0.0):>6.1f}% | {m.get('top5_acc_pct', 0.0):>6.1f}% | {m.get('valid_smiles_rate_pct', 0.0):>6.1f}% | {m.get('ece', 0.0):>6.4f} | {m.get('median_latency_ms', 0.0):>8.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
