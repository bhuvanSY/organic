"""
Unit tests for ChemRxn-Intelligence Evaluation Framework.

Tests:
- Chemical syntax failure taxonomy (syntax vs valence vs kekulization vs valid)
- Stereochemistry-aware vs constitutional exact match
- Bootstrap confidence interval estimation
- Expected calibration error (ECE)
- Latency and percentile distributions
- Scaffold generalization & leakage metrics
- EvaluationReporter exports (JSON, CSV, Markdown, Plots)
"""

import json
from pathlib import Path
import pytest
import numpy as np

from chemrxn.evaluation.metrics import (
    categorize_smiles_failure,
    compute_validity_breakdown,
    compute_stereo_aware_exact_match,
    compute_exact_match_accuracy,
    compute_bootstrap_confidence_interval,
    compute_expected_calibration_error,
    compute_latency_distribution,
    extract_murcko_scaffold_smiles,
    evaluate_scaffold_generalization,
    SCIENTIFIC_DISCLAIMER,
)
from chemrxn.evaluation.reporter import EvaluationReporter


def test_categorize_smiles_failure():
    # Valid
    assert categorize_smiles_failure("CC(=O)O") == "VALID"
    assert categorize_smiles_failure("c1ccccc1") == "VALID"

    # Empty
    assert categorize_smiles_failure("") == "EMPTY_OR_WHITESPACE"
    assert categorize_smiles_failure("   ") == "EMPTY_OR_WHITESPACE"

    # Syntax error (unclosed ring or unmatched paren)
    assert categorize_smiles_failure("CC(=O") == "SYNTAX_ERROR"
    assert categorize_smiles_failure("c1ccccc") == "SYNTAX_ERROR"

    # Valence violation (hypervalent carbon)
    assert categorize_smiles_failure("C(C)(C)(C)(C)C") == "VALENCE_VIOLATION"


def test_compute_validity_breakdown():
    smiles_list = [
        "CC(=O)O",          # valid
        "c1ccccc1",         # valid
        "C(C)(C)(C)(C)C",   # valence violation
        "CC(=O",            # syntax error
        "",                 # empty
    ]
    res = compute_validity_breakdown(smiles_list)
    assert res["total_evaluated"] == 5
    assert res["valid_rate"] == 0.4
    assert res["categories"]["VALENCE_VIOLATION"] == 1
    assert res["categories"]["SYNTAX_ERROR"] == 1
    assert res["categories"]["EMPTY_OR_WHITESPACE"] == 1
    assert res["categories"]["VALID"] == 2


def test_compute_stereo_aware_exact_match():
    # Exact stereo match: L-alanine
    smi1 = "C[C@H](N)C(=O)O"
    smi2 = "N[C@@H](C)C(=O)O"  # Equivalent enantiomer representation
    stereo_match, non_stereo_match = compute_stereo_aware_exact_match(smi1, smi2)
    assert stereo_match is True
    assert non_stereo_match is True

    # Inverted stereo: L-alanine vs D-alanine
    smi_d = "C[C@@H](N)C(=O)O"
    stereo_match_d, non_stereo_match_d = compute_stereo_aware_exact_match(smi1, smi_d)
    assert stereo_match_d is False
    assert non_stereo_match_d is True  # Same constitutional formula and connectivity

    # Completely different molecule
    stereo_diff, non_stereo_diff = compute_stereo_aware_exact_match(smi1, "c1ccccc1")
    assert stereo_diff is False
    assert non_stereo_diff is False


def test_compute_exact_match_accuracy():
    preds = [
        ["CC(=O)O", "CCO", "c1ccccc1"],
        ["CCO", "CCN", "CCC"],
        ["c1ccccc1", "c1ccncc1"],
    ]
    targets = [
        "CC(=O)O",   # in top-1, top-3, top-5
        "CCC",       # in top-3, top-5 (rank 3)
        "c1ccccc1O", # not in candidates
    ]
    top1 = compute_exact_match_accuracy(preds, targets, top_k=1)
    top3 = compute_exact_match_accuracy(preds, targets, top_k=3)
    top5 = compute_exact_match_accuracy(preds, targets, top_k=5)
    assert top1 == pytest.approx(1.0 / 3.0)
    assert top3 == pytest.approx(2.0 / 3.0)
    assert top5 == pytest.approx(2.0 / 3.0)


def test_compute_bootstrap_confidence_interval():
    data = [1.0] * 80 + [0.0] * 20
    ci = compute_bootstrap_confidence_interval(data, n_bootstrap=500, ci=0.95, seed=42)
    assert ci["mean"] == pytest.approx(0.80)
    assert ci["ci_lower"] <= ci["mean"]
    assert ci["ci_upper"] >= ci["mean"]
    assert 0.70 <= ci["ci_lower"] <= 0.80
    assert 0.80 <= ci["ci_upper"] <= 0.90


def test_compute_expected_calibration_error():
    # Perfectly calibrated predictions
    confs = [0.9, 0.9, 0.1, 0.1]
    correct = [1.0, 1.0, 0.0, 0.0]
    res = compute_expected_calibration_error(confs, correct, n_bins=5)
    assert "ece" in res
    assert res["ece"] < 0.15

    # Completely miscalibrated (100% confidence, 0% accuracy)
    bad_confs = [1.0, 1.0, 1.0, 1.0]
    bad_correct = [0.0, 0.0, 0.0, 0.0]
    bad_res = compute_expected_calibration_error(bad_confs, bad_correct, n_bins=5)
    assert bad_res["ece"] == pytest.approx(1.0)


def test_compute_latency_distribution():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    dist = compute_latency_distribution(latencies)
    assert dist["mean_ms"] == pytest.approx(55.0)
    assert dist["median_ms"] == pytest.approx(55.0)
    assert dist["p90_ms"] >= 90.0
    assert dist["throughput_rxn_per_sec"] > 0


def test_scaffold_metrics_and_leakage():
    # Bemis-Murcko extraction
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    scaff = extract_murcko_scaffold_smiles(aspirin)
    assert scaff == "c1ccccc1"

    # Scaffold generalization and leakage test
    test_rxns = [
        ("CC(=O)Cl.c1ccccc1", "CC(=O)c1ccccc1"),  # scaffold: c1ccccc1
        ("CCN.CCO", "CCN(CC)CC"),                 # aliphatic: empty scaffold
    ]
    preds = [
        "CC(=O)c1ccccc1",  # correct
        "CCN(CC)CC",       # correct
    ]
    train_products = ["c1ccccc1O", "c1ccccc1"]  # c1ccccc1 is in train!

    res = evaluate_scaffold_generalization(test_rxns, preds, train_products)
    assert res["test_reactions_evaluated"] == 2
    assert res["seen_scaffold_samples"] == 1
    assert res["unseen_scaffold_samples"] == 1
    assert res["seen_scaffold_accuracy"] == 100.0


def test_evaluation_reporter(tmp_path):
    reporter = EvaluationReporter(output_dir=str(tmp_path))

    mock_summaries = [
        {
            "model_name": "Seq2Seq Transformer",
            "top1_acc_pct": 65.0,
            "top1_ci_lower": 58.0,
            "top1_ci_upper": 72.0,
            "top3_acc_pct": 82.0,
            "top5_acc_pct": 88.0,
            "stereo_aware_acc_pct": 63.0,
            "valid_smiles_rate_pct": 98.0,
            "abstention_rate_pct": 0.0,
            "precision_at_coverage_pct": 65.0,
            "ece": 0.045,
            "median_latency_ms": 12.5,
            "throughput_rxn_sec": 80.0,
            "parameter_count": "12,450,000",
        },
        {
            "model_name": "Hybrid Inference Engine",
            "top1_acc_pct": 74.0,
            "top1_ci_lower": 68.0,
            "top1_ci_upper": 80.0,
            "top3_acc_pct": 89.0,
            "top5_acc_pct": 93.0,
            "stereo_aware_acc_pct": 72.0,
            "valid_smiles_rate_pct": 100.0,
            "abstention_rate_pct": 5.0,
            "precision_at_coverage_pct": 77.8,
            "ece": 0.021,
            "median_latency_ms": 16.2,
            "throughput_rxn_sec": 61.7,
            "parameter_count": "13,100,000",
        }
    ]

    mock_results = {
        "metadata": {
            "evaluator_version": "1.0.0",
            "test_samples_evaluated": 100,
            "hardware_device": "cuda:0",
            "abstention_threshold": 0.35,
            "disclaimer": SCIENTIFIC_DISCLAIMER,
        },
        "model_summaries": mock_summaries,
        "validity_breakdown": {"total_predictions": 100, "valid_count": 99},
        "scaffold_generalization": {"exact_product_leakage_count": 0},
    }

    # Test JSON export
    json_path = reporter.export_json(mock_results, "test_eval.json")
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["metadata"]["test_samples_evaluated"] == 100

    # Test CSV export
    csv_path = reporter.export_csv(mock_summaries, "test_eval.csv")
    assert csv_path.exists()

    # Test Plots export
    plots = reporter.generate_plots(mock_summaries)
    assert "accuracy_plot" in plots
    assert Path(plots["accuracy_plot"]).exists()
    assert "latency_accuracy_plot" in plots
    assert Path(plots["latency_accuracy_plot"]).exists()

    # Test Markdown report
    md_path = reporter.generate_markdown_report(mock_results, mock_summaries, "test_report.md")
    assert md_path.exists()
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Rigorous Scientific Evaluation Report" in content
        assert "Hybrid Inference Engine" in content
