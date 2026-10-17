"""
Deterministic dataset splitting and leakage auditing.
Supports scaffold-aware grouping (Bemis-Murcko) and rigorous cross-split overlap detection.
"""

from collections import defaultdict
import random
from typing import Any, Dict, List, Set, Tuple
import numpy as np

from chemrxn.data.schemas import ProcessedReaction


def deterministic_random_split(
    records: List[ProcessedReaction],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[ProcessedReaction], List[ProcessedReaction], List[ProcessedReaction]]:
    """
    Deterministically partitions records using a pseudo-random shuffle with fixed seed.
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = shuffled[:train_end]
    val_set = shuffled[train_end:val_end]
    test_set = shuffled[val_end:]

    return train_set, val_set, test_set


def scaffold_split(
    records: List[ProcessedReaction],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[ProcessedReaction], List[ProcessedReaction], List[ProcessedReaction]]:
    """
    Scaffold-aware splitting grouping by product Murcko scaffold.
    Prevents structural product leakage into validation and test sets.
    Uses balanced bin-packing with deterministic tie-breaking.
    """
    total = len(records)
    target_train = int(total * train_ratio)
    target_val = int(total * val_ratio)
    target_test = total - target_train - target_val

    # Group reactions by scaffold
    scaffold_groups: Dict[str, List[ProcessedReaction]] = defaultdict(list)
    for r in records:
        scaff = r.product_scaffold or "_unknown_"
        scaffold_groups[scaff].append(r)

    # Deterministic shuffle within groups
    rng = random.Random(seed)
    for scaff in scaffold_groups:
        rng.shuffle(scaffold_groups[scaff])

    # Sort scaffold groups descending by group size, tie-breaking by scaffold hash
    sorted_scaffolds = sorted(
        scaffold_groups.keys(),
        key=lambda s: (len(scaffold_groups[s]), hash(s)),
        reverse=True,
    )

    train_records: List[ProcessedReaction] = []
    val_records: List[ProcessedReaction] = []
    test_records: List[ProcessedReaction] = []

    for scaff in sorted_scaffolds:
        group = scaffold_groups[scaff]
        group_len = len(group)

        # Allocate to the split that has the greatest relative deficit
        train_deficit = max(0, target_train - len(train_records))
        val_deficit = max(0, target_val - len(val_records))
        test_deficit = max(0, target_test - len(test_records))

        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_records.extend(group)
        elif val_deficit >= test_deficit:
            val_records.extend(group)
        else:
            test_records.extend(group)

    return train_records, val_records, test_records


def audit_split_leakage(
    train: List[ProcessedReaction],
    val: List[ProcessedReaction],
    test: List[ProcessedReaction],
) -> Dict[str, Any]:
    """
    Rigorously checks for duplicate reactions and scaffold leakage across splits.
    """
    def to_reaction_keys(recs: List[ProcessedReaction]) -> Set[str]:
        return {f"{r.reactants_smiles}>>{r.products_smiles}" for r in recs}

    def to_product_keys(recs: List[ProcessedReaction]) -> Set[str]:
        return {r.products_smiles for r in recs}

    def to_scaffold_keys(recs: List[ProcessedReaction]) -> Set[str]:
        return {r.product_scaffold for r in recs if r.product_scaffold not in ("", "_unknown_", "_acyclic_")}

    train_rxns = to_reaction_keys(train)
    val_rxns = to_reaction_keys(val)
    test_rxns = to_reaction_keys(test)

    train_val_rxn_overlap = train_rxns.intersection(val_rxns)
    train_test_rxn_overlap = train_rxns.intersection(test_rxns)
    val_test_rxn_overlap = val_rxns.intersection(test_rxns)

    train_products = to_product_keys(train)
    test_products = to_product_keys(test)
    product_overlap = train_products.intersection(test_products)

    train_scaffolds = to_scaffold_keys(train)
    test_scaffolds = to_scaffold_keys(test)
    scaffold_overlap = train_scaffolds.intersection(test_scaffolds)

    return {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "exact_reaction_leakage": {
            "train_val_overlap_count": len(train_val_rxn_overlap),
            "train_test_overlap_count": len(train_test_rxn_overlap),
            "val_test_overlap_count": len(val_test_rxn_overlap),
            "total_reaction_leakage": len(train_val_rxn_overlap) + len(train_test_rxn_overlap) + len(val_test_rxn_overlap),
        },
        "product_molecule_leakage": {
            "train_test_product_overlap_count": len(product_overlap),
            "train_test_product_overlap_rate": round(len(product_overlap) / max(1, len(test_products)), 4),
        },
        "scaffold_overlap": {
            "train_scaffolds_count": len(train_scaffolds),
            "test_scaffolds_count": len(test_scaffolds),
            "train_test_scaffold_overlap_count": len(scaffold_overlap),
            "train_test_scaffold_overlap_rate": round(len(scaffold_overlap) / max(1, len(test_scaffolds)), 4),
        },
    }
