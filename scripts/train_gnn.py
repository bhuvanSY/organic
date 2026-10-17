"""
Standalone training script for the Molecular Reaction GNN.
Supports:
- CLI parameters for dataset, batch size, learning rate, and epochs
- --overfit-test flag for immediate convergence verification in seconds
- In-batch negative sampling for candidate ranking
- Checkpointing and JSON training telemetry report
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple
import pandas as pd
import torch

from chemrxn.chemistry.graphs import detect_reaction_centers, mol_to_graph
from chemrxn.models.molecular_gnn import MolecularReactionGNN
from chemrxn.training.gnn_trainer import GNNReactionTrainer, collate_gnn_reaction_batch, GNNReactionBatch


TOY_OVERFIT_REACTIONS = [
    ("C=CCBr.CC(C#N)c1ccccc1", "C=CCC(C)(C#N)c1ccccc1", "CC(C#N)c1ccccc1"),
    ("CC(=O)Cl.CCN", "CCNC(=O)C", "CCNC(=O)CC"),
    ("COC(=O)c1ccc(OC)c(F)c1", "COc1ccc(C(=O)O)cc1F", "COC(=O)c1ccccc1"),
    ("c1ccc(C(=O)Cl)cc1.O", "O=C(O)c1ccccc1", "O=C(O)c1cccc(Cl)c1"),
    ("CC(C)N.CC(=O)Cl", "CC(=O)NC(C)C", "CC(=O)c1ccccc1"),
    ("CCO.CC(=O)O", "CCOC(=O)C", "CCOCCOC"),
    ("Brc1ccccc1.c1ccccc1B(O)O", "c1ccc(-c2ccccc2)cc1", "c1ccccc1"),
    ("CC(=O)c1ccccc1.O=C(O)O", "CC(=O)c1ccccc1O", "CC(=O)c1ccccc1"),
    ("CCN.CC(C)=O", "CCNC(C)C", "CC(C)O"),
    ("c1ccccc1O.CC(=O)Cl", "CC(=O)Oc1ccccc1", "c1ccccc1OC(C)C"),
    ("CC#N.[Li]C", "CC(=O)C", "CC(C)N"),
    ("CC(=O)Nc1ccccc1.BrCC", "CCN(C(=O)C)c1ccccc1", "CC(=O)Nc1ccccc1"),
    ("c1ccccc1C(=O)O.CN", "CNC(=O)c1ccccc1", "CNCc1ccccc1"),
    ("c1ccccc1Br.CCNCC", "CCN(CC)c1ccccc1", "c1ccccc1N"),
    ("CC(C)(C)OC(=O)NC.Cl", "CNC", "CNC(=O)OC(C)(C)C"),
]


def load_dataset_samples(path: str, max_samples: int = 1000) -> List[Tuple[str, str]]:
    """Loads reactant and product SMILES pairs from CSV or Parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, nrows=max_samples * 2)

    samples: List[Tuple[str, str]] = []
    r_col = "reactants_smiles" if "reactants_smiles" in df.columns else "reactants"
    p_col = "products_smiles" if "products_smiles" in df.columns else "products"

    for _, row in df.iterrows():
        r = str(row[r_col]).strip()
        p = str(row[p_col]).strip()
        if r and p and r != "nan" and p != "nan":
            samples.append((r, p))
            if len(samples) >= max_samples:
                break

    return samples


def create_batches(
    pairs: List[Tuple[str, str]],
    batch_size: int = 32,
) -> List[GNNReactionBatch]:
    """
    Creates GNNReactionBatches from reactant-product pairs using in-batch negative product sampling.
    """
    batches: List[GNNReactionBatch] = []
    n = len(pairs)
    if n == 0:
        return batches

    for i in range(0, n, batch_size):
        chunk = pairs[i : i + batch_size]
        if len(chunk) < 2 and n >= 2:
            # Need at least 2 samples for in-batch negative sampling
            continue

        sample_triplets: List[Tuple[str, str, str]] = []
        for j, (r, p) in enumerate(chunk):
            # Select negative product from another reaction in chunk
            neg_idx = (j + 1) % len(chunk)
            neg_p = chunk[neg_idx][1]
            sample_triplets.append((r, p, neg_p))

        batch = collate_gnn_reaction_batch(sample_triplets)
        if batch is not None:
            batches.append(batch)

    return batches


def main():
    parser = argparse.ArgumentParser(description="Train Molecular Reaction GNN from scratch.")
    parser.add_argument("--train-path", type=str, default="data/processed/train.parquet")
    parser.add_argument("--valid-path", type=str, default="data/processed/valid.parquet")
    parser.add_argument("--max-samples", type=str, default="1000", help="Max samples to load or 'all'")
    parser.add_argument("--overfit-test", action="store_true", help="Run 15-sample overfit verification")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gnn")
    parser.add_argument("--report-path", type=str, default="reports/gnn_training_report.json")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ChemRxn GNN] Training on device: {device}")

    # Build Model
    model = MolecularReactionGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    param_count = model.count_parameters()
    print(f"[ChemRxn GNN] Randomly initialized GNN with {param_count:,} trainable parameters.")

    trainer = GNNReactionTrainer(
        model=model,
        learning_rate=args.lr,
        device=device,
        use_amp=not args.no_amp,
    )

    if args.overfit_test:
        print("[ChemRxn GNN] === Running Fast Overfit Verification (15 reactions) ===")
        triplets = TOY_OVERFIT_REACTIONS
        train_batches = [collate_gnn_reaction_batch(triplets)]
        val_batches = train_batches
        epochs = 60
    else:
        # Check CSV fallback if parquet doesn't exist
        train_path = args.train_path
        if not os.path.exists(train_path) and os.path.exists("data/processed/train.csv"):
            train_path = "data/processed/train.csv"
        valid_path = args.valid_path
        if not os.path.exists(valid_path) and os.path.exists("data/processed/valid.csv"):
            valid_path = "data/processed/valid.csv"

        max_s = None if args.max_samples == "all" else int(args.max_samples)
        train_pairs = load_dataset_samples(train_path, max_samples=max_s or 5000)
        val_pairs = load_dataset_samples(valid_path, max_samples=min(max_s or 5000, 500))

        print(f"[ChemRxn GNN] Loaded {len(train_pairs)} train pairs, {len(val_pairs)} val pairs.")
        train_batches = create_batches(train_pairs, batch_size=args.batch_size)
        val_batches = create_batches(val_pairs, batch_size=args.batch_size)
        epochs = args.epochs

    print(f"[ChemRxn GNN] Starting training for {epochs} epochs across {len(train_batches)} batches...")
    start_time = time.time()
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_metrics = trainer.train_epoch(train_batches)
        val_metrics = trainer.evaluate(val_batches)
        ep_duration = time.time() - ep_start

        log_data = {
            "epoch": epoch,
            "duration_sec": round(ep_duration, 2),
            **{k: round(v, 4) for k, v in train_metrics.items()},
            **{k: round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(log_data)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.2f}s] | "
                f"Train Loss: {train_metrics['loss']:.4f} (Rank Acc: {train_metrics['rank_accuracy']*100:.1f}%, RC Acc: {train_metrics['rc_atom_accuracy']*100:.1f}%) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} (Rank Acc: {val_metrics['val_rank_accuracy']*100:.1f}%)"
            )

        # Save best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_ckpt_path = os.path.join(args.checkpoint_dir, "best_gnn.pt")
            model.save_checkpoint(best_ckpt_path, extra_meta={"epoch": epoch, "val_metrics": val_metrics})

    total_time = time.time() - start_time
    gpu_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    )

    print(f"[ChemRxn GNN] Finished in {total_time:.2f}s! Best Val Loss: {best_val_loss:.4f} | Max GPU VRAM: {gpu_mem:.2f} MB")

    # Save report
    report = {
        "status": "success",
        "overfit_test": args.overfit_test,
        "trainable_parameters": param_count,
        "total_runtime_seconds": round(total_time, 2),
        "gpu_memory_allocated_mb": round(gpu_mem, 2),
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[ChemRxn GNN] Telemetry report saved to {args.report_path}")


if __name__ == "__main__":
    main()
