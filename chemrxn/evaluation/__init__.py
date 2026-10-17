"""Evaluation metrics and benchmark suites for ChemRxn-Intelligence."""
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
from chemrxn.evaluation.benchmark_runner import UnifiedBenchmarkSuite, load_locked_test_dataset


def compute_validity_rate(smiles_list):
    """Backward-compatible helper returning valid fraction."""
    return compute_validity_breakdown(smiles_list)["valid_rate"]


__all__ = [
    "categorize_smiles_failure",
    "compute_validity_breakdown",
    "compute_validity_rate",
    "compute_stereo_aware_exact_match",
    "compute_exact_match_accuracy",
    "compute_bootstrap_confidence_interval",
    "compute_expected_calibration_error",
    "compute_latency_distribution",
    "extract_murcko_scaffold_smiles",
    "evaluate_scaffold_generalization",
    "SCIENTIFIC_DISCLAIMER",
    "EvaluationReporter",
    "UnifiedBenchmarkSuite",
    "load_locked_test_dataset",
]
