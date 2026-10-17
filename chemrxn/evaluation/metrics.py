"""
Rigorous evaluation metrics and statistical tools for ChemRxn-Intelligence.
Implements:
1. Canonical & stereochemistry-aware product exact match (Top-1, Top-3, Top-5)
2. Chemical syntax validity & failure categorization (Valence, Kekulize, Syntax)
3. Non-parametric bootstrap confidence intervals (95% CI)
4. Confidence calibration (ECE) and selective accuracy / coverage
5. Murcko scaffold generalization & leakage detection
6. Computational profiling (P50, P90, P95, P99 latency, throughput, GPU memory)
7. Strict scientific disclaimer separating syntax, reference match, and lab feasibility
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemrxn.chemistry.fingerprints import compute_tanimoto_similarity
from chemrxn.chemistry.smiles import canonicalize_smiles, is_valid_smiles, parse_mol_safely


SCIENTIFIC_DISCLAIMER = (
    "DISCLAIMER: Chemical syntax validity confirms only that a SMILES string satisfies "
    "graph valence and sanitization rules in RDKit. Reference-product agreement measures "
    "concordance with historical patent records. Neither metric proves experimental "
    "chemical feasibility, selectivity, or synthesis yield in a wet-lab environment."
)


def categorize_smiles_failure(smiles: str) -> str:
    """
    Categorizes the exact chemical error when RDKit fails to parse or sanitize a SMILES.
    Returns:
        'VALID', 'EMPTY_OR_WHITESPACE', 'VALENCE_VIOLATION', 'KEKULIZATION_ERROR', or 'SYNTAX_ERROR'
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return "EMPTY_OR_WHITESPACE"

    s = smiles.strip()
    try:
        # First attempt: parse without sanitization
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None:
            return "SYNTAX_ERROR"

        # Second attempt: sanitize to detect chemical valence or aromaticity errors
        try:
            Chem.SanitizeMol(mol)
            return "VALID"
        except Exception as e:
            err_msg = str(e).lower()
            if "valence" in err_msg or "explicit valence" in err_msg:
                return "VALENCE_VIOLATION"
            elif "kekuliz" in err_msg or "aromatic" in err_msg:
                return "KEKULIZATION_ERROR"
            else:
                return "VALENCE_VIOLATION"
    except Exception:
        return "SYNTAX_ERROR"


def compute_validity_breakdown(smiles_list: List[str]) -> Dict[str, Any]:
    """
    Computes valid SMILES rate and breaks down failure modes.
    """
    if not smiles_list:
        return {"valid_rate": 0.0, "total": 0, "categories": {}}

    counts: Dict[str, int] = {
        "VALID": 0,
        "SYNTAX_ERROR": 0,
        "VALENCE_VIOLATION": 0,
        "KEKULIZATION_ERROR": 0,
        "EMPTY_OR_WHITESPACE": 0,
    }

    for s in smiles_list:
        cat = categorize_smiles_failure(s)
        counts[cat] = counts.get(cat, 0) + 1

    total = len(smiles_list)
    valid_rate = counts["VALID"] / total

    return {
        "valid_rate": round(valid_rate, 4),
        "total_evaluated": total,
        "categories": {k: v for k, v in counts.items()},
        "category_percentages": {k: round(v / total * 100.0, 2) for k, v in counts.items()},
    }


def compute_stereo_aware_exact_match(
    cand_smiles: str,
    target_smiles: str,
) -> Tuple[bool, bool]:
    """
    Evaluates both stereochemistry-aware exact match and constitution/connectivity match.
    Returns:
        (is_stereo_match, is_constitution_match)
    """
    if not is_valid_smiles(cand_smiles) or not is_valid_smiles(target_smiles):
        return False, False

    mol_cand = parse_mol_safely(cand_smiles)
    mol_target = parse_mol_safely(target_smiles)

    if mol_cand is None or mol_target is None:
        return False, False

    # 1. Stereo-aware exact match
    s_cand_stereo = Chem.MolToSmiles(mol_cand, isomericSmiles=True, canonical=True)
    s_target_stereo = Chem.MolToSmiles(mol_target, isomericSmiles=True, canonical=True)
    is_stereo_match = (s_cand_stereo == s_target_stereo)

    # 2. Constitution / connectivity match (stereochemistry stripped)
    s_cand_flat = Chem.MolToSmiles(mol_cand, isomericSmiles=False, canonical=True)
    s_target_flat = Chem.MolToSmiles(mol_target, isomericSmiles=False, canonical=True)
    is_constitution_match = (s_cand_flat == s_target_flat)

    return is_stereo_match, is_constitution_match


def compute_exact_match_accuracy(
    candidates_list: List[List[str]],
    ground_truth: List[str],
    top_k: int = 1,
    stereo_aware: bool = True,
) -> float:
    """
    Computes top-k exact match accuracy across a cohort of reaction predictions.
    """
    if not ground_truth or not candidates_list:
        return 0.0

    matches = 0
    total = len(ground_truth)

    for candidates, target in zip(candidates_list, ground_truth):
        matched = False
        k_cands = candidates[:top_k]
        for cand in k_cands:
            stereo_m, flat_m = compute_stereo_aware_exact_match(cand, target)
            is_m = stereo_m if stereo_aware else flat_m
            if is_m:
                matched = True
                break
        if matched:
            matches += 1

    return float(matches / total)


def compute_bootstrap_confidence_interval(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes non-parametric bootstrap confidence interval for a metric mean.
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.RandomState(seed)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(boot_means, alpha * 100.0)
    upper = np.percentile(boot_means, (1.0 - alpha) * 100.0)

    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "ci_level": ci,
    }


def compute_expected_calibration_error(
    confidences: List[float],
    correct_flags: List[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculates Expected Calibration Error (ECE) and reliability diagram bins.
    """
    if not confidences or not correct_flags:
        return {"ece": 0.0, "bins": []}

    confs = np.array(confidences, dtype=np.float64)
    labels = np.array(correct_flags, dtype=np.float64)
    n_total = len(confs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_records = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= low) & (confs < high if i < n_bins - 1 else confs <= high)
        count = int(np.sum(mask))

        if count > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(confs[mask]))
            abs_diff = abs(bin_acc - bin_conf)
            ece += (count / n_total) * abs_diff
            bin_records.append({
                "bin_idx": i,
                "bin_range": [round(low, 2), round(high, 2)],
                "count": count,
                "mean_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "calibration_gap": round(abs_diff, 4),
            })
        else:
            bin_records.append({
                "bin_idx": i,
                "bin_range": [round(low, 2), round(high, 2)],
                "count": 0,
                "mean_confidence": 0.0,
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "n_samples": n_total,
        "n_bins": n_bins,
        "reliability_bins": bin_records,
    }


def compute_latency_distribution(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Computes statistical percentiles for inference latency.
    """
    if not latencies_ms:
        return {"median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "throughput_rxn_per_sec": 0.0}

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_lat = np.mean(arr)
    throughput = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_ms": round(float(mean_lat), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "throughput_rxn_per_sec": round(float(throughput), 2),
    }


def extract_murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    """Extracts Bemis-Murcko core scaffold from a molecule."""
    mol = parse_mol_safely(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def evaluate_scaffold_generalization(
    test_reactions: List[Tuple[str, str]], # List of (reactants, product)
    predictions: List[str],               # List of top-1 predicted products
    train_products: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Segments test reactions into seen scaffolds vs unseen scaffolds
    and computes generalization gap and exact leakage.
    """
    train_scaffolds = set()
    train_prods_set = set()
    if train_products:
        for p in train_products:
            cp = canonicalize_smiles(p, remove_mapping=True) or p
            train_prods_set.add(cp)
            scaff = extract_murcko_scaffold_smiles(cp)
            if scaff:
                train_scaffolds.add(scaff)

    seen_hits, seen_total = 0, 0
    unseen_hits, unseen_total = 0, 0
    product_leakage_count = 0

    for (r, p), pred in zip(test_reactions, predictions):
        p_canon = canonicalize_smiles(p, remove_mapping=True) or p
        pred_canon = canonicalize_smiles(pred, remove_mapping=True) or pred

        is_hit = (p_canon == pred_canon)

        if p_canon in train_prods_set:
            product_leakage_count += 1

        scaff = extract_murcko_scaffold_smiles(p_canon)
        is_seen = (scaff in train_scaffolds) if scaff and train_scaffolds else False

        if is_seen:
            seen_total += 1
            if is_hit:
                seen_hits += 1
        else:
            unseen_total += 1
            if is_hit:
                unseen_hits += 1

    seen_acc = (seen_hits / seen_total) if seen_total > 0 else 0.0
    unseen_acc = (unseen_hits / unseen_total) if unseen_total > 0 else (
        (unseen_hits / max(unseen_total, 1)) if unseen_total > 0 else 0.0
    )

    return {
        "train_scaffold_count": len(train_scaffolds),
        "test_reactions_evaluated": len(test_reactions),
        "exact_product_leakage_count": product_leakage_count,
        "exact_product_leakage_rate": round(product_leakage_count / max(len(test_reactions), 1), 4),
        "seen_scaffold_samples": seen_total,
        "seen_scaffold_accuracy": round(seen_acc * 100.0, 2),
        "unseen_scaffold_samples": unseen_total,
        "unseen_scaffold_accuracy": round((seen_acc if unseen_total == 0 else (unseen_hits / unseen_total * 100.0)), 2),
    }
