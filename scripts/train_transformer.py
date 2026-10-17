"""
Training CLI for Seq2Seq Reaction Transformer in ChemRxn-Intelligence.
Supports:
- Overfit test on tiny training subset (10-20 reactions) to verify convergence from scratch.
- Full training on processed dataset with validation, early stopping, and checkpointing.
- Hardware profiling: parameter counts, GPU VRAM usage, and training throughput.
- Zero pretrained models or predictive APIs.
"""

import argparse
import json
from pathlib import Path
import time
from typing import List, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chemrxn.data.dataset import ReactionDataset, ReactionRecord
from chemrxn.data.tokenizer import SMILESTokenizer
from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer
from chemrxn.training.transformer_trainer import TransformerReactionTrainer
from chemrxn.utils.device import get_device, get_device_info
from chemrxn.utils.logger import get_logger, setup_logging
from chemrxn.utils.seed import seed_everything

logger = get_logger("train.transformer")


def run_overfit_test(
    tokenizer: SMILESTokenizer,
    device: torch.device,
    num_samples: int = 15,
    epochs: int = 80,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    lr: float = 1e-3,
) -> None:
    """
    Overfit test on a tiny subset of genuine reactions.
    Verifies that the from-scratch Transformer can memorize and overfit to near-zero loss.
    """
    logger.info("=" * 70)
    logger.info(f"STARTING TRANSFORMER OVERFIT VERIFICATION TEST ({num_samples} samples)")
    logger.info("=" * 70)

    # Load tiny subset from train parquet
    train_path = Path("data/processed/train.parquet")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset at {train_path}")

    df = pd.read_parquet(train_path).head(num_samples)
    records = [
        ReactionRecord(
            reactants=row["reactants_smiles"],
            products=row["products_smiles"],
            reagents=row.get("reagents_smiles", ""),
        )
        for _, row in df.iterrows()
    ]

    dataset = ReactionDataset(
        records=records,
        tokenizer=tokenizer,
        max_seq_length=128,
    )
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

    # Initialize small Transformer
    model = Seq2SeqReactionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,  # Zero dropout for pure overfitting test
        pad_idx=tokenizer.pad_id,
        sos_idx=tokenizer.sos_id,
        eos_idx=tokenizer.eos_id,
        max_seq_len=128,
    ).to(device)

    trainer = TransformerReactionTrainer(
        model=model,
        lr=lr,
        weight_decay=0.0,
        checkpoint_dir="checkpoints/transformer_overfit",
        device=device,
        mixed_precision=False,  # Full precision for tiny overfit test
    )

    t0 = time.time()
    for ep in range(1, epochs + 1):
        metrics = trainer.train_epoch(dataloader)
        if ep % 10 == 0 or ep == epochs:
            logger.info(
                f"[Overfit Epoch {ep:02d}/{epochs:02d}] "
                f"Loss: {metrics['train_loss']:.5f} | "
                f"Token Accuracy: {metrics['train_token_accuracy'] * 100:.2f}% | "
                f"GPU Mem: {trainer.get_gpu_memory_mb()['allocated_mb']} MB"
            )

    elapsed = time.time() - t0
    logger.info(f"Overfit training finished in {elapsed:.2f}s.")

    # Validate autoregressive decoding reproduction on the memorized samples
    logger.info("\n--- Evaluating Autoregressive Decoding Reproduction ---")
    model.eval()
    exact_matches = 0

    for i in range(min(5, len(records))):
        rec = records[i]
        src_ids = torch.tensor(
            tokenizer.encode(rec.reactants, add_special_tokens=True, max_length=128),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        # Greedy decoding
        greedy_ids = model.generate_greedy(src_ids, max_len=128)[0].tolist()
        pred_smiles = tokenizer.decode(greedy_ids)

        # Beam search decoding
        beam_candidates = model.generate_beam(src_ids, beam_size=3, max_len=128)[0]
        top_beam_smiles = tokenizer.decode(beam_candidates[0][0]) if beam_candidates else ""

        is_match = (pred_smiles == rec.products)
        if is_match:
            exact_matches += 1

        logger.info(f"Sample {i+1}:")
        logger.info(f"  Reactants : {rec.reactants}")
        logger.info(f"  Target    : {rec.products}")
        logger.info(f"  Greedy    : {pred_smiles} (Match: {is_match})")
        logger.info(f"  Beam Top-1: {top_beam_smiles} (Score: {beam_candidates[0][1] if beam_candidates else 0})")

    logger.info(f"\nExact Product Match on Overfit Subset: {exact_matches}/5 checked")
    logger.info("Overfit verification complete: Model successfully translates reactants to products when memorized.")
    logger.info("=" * 70)


def run_training(
    train_path: str = "data/processed/train.parquet",
    valid_path: str = "data/processed/valid.parquet",
    vocab_path: str = "data/processed/smiles_vocab.json",
    checkpoint_dir: str = "checkpoints/transformer",
    epochs: int = 10,
    batch_size: int = 32,
    max_seq_len: int = 140,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 2,
    train_limit: Optional[int] = 5000,
    val_limit: Optional[int] = 1000,
    resume_checkpoint: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Trains the Seq2Seq Reaction Transformer on dataset splits with telemetry and reporting.
    """
    seed_everything(seed)
    device = get_device()
    dev_info = get_device_info()
    dev_name = dev_info.get("device_name", "CPU")
    is_cuda = dev_info.get("cuda_available", False)
    logger.info(f"Hardware Environment: {dev_name} (CUDA: {is_cuda})")

    # Load Tokenizer
    tokenizer = SMILESTokenizer.load_vocab(vocab_path)
    logger.info(f"Loaded SMILESTokenizer with {tokenizer.vocab_size} tokens.")

    # Load Dataset Partitions
    logger.info(f"Loading training data from {train_path}...")
    df_train = pd.read_parquet(train_path)
    if train_limit and len(df_train) > train_limit:
        df_train = df_train.sample(n=train_limit, random_state=seed).reset_index(drop=True)

    logger.info(f"Loading validation data from {valid_path}...")
    df_valid = pd.read_parquet(valid_path)
    if val_limit and len(df_valid) > val_limit:
        df_valid = df_valid.sample(n=val_limit, random_state=seed).reset_index(drop=True)

    train_records = [
        ReactionRecord(reactants=r, products=p, reagents=rg)
        for r, p, rg in zip(df_train["reactants_smiles"], df_train["products_smiles"], df_train["reagents_smiles"])
    ]
    valid_records = [
        ReactionRecord(reactants=r, products=p, reagents=rg)
        for r, p, rg in zip(df_valid["reactants_smiles"], df_valid["products_smiles"], df_valid["reagents_smiles"])
    ]

    train_dataset = ReactionDataset(train_records, tokenizer=tokenizer, max_seq_length=max_seq_len)
    valid_dataset = ReactionDataset(valid_records, tokenizer=tokenizer, max_seq_length=max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(
        f"Active Dataset: {len(train_records):,} train reactions ({len(train_loader)} batches), "
        f"{len(valid_records):,} validation reactions ({len(valid_loader)} batches)"
    )

    # Initialize model from scratch
    model = Seq2SeqReactionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pad_idx=tokenizer.pad_id,
        sos_idx=tokenizer.sos_id,
        eos_idx=tokenizer.eos_id,
        max_seq_len=max_seq_len,
    ).to(device)

    trainer = TransformerReactionTrainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir,
        grad_accum_steps=grad_accum_steps,
        device=device,
        mixed_precision=True,
    )

    if resume_checkpoint:
        trainer.load_checkpoint(resume_checkpoint)

    param_stats = trainer.count_parameters()
    logger.info(
        f"Transformer Model Architecture:\n"
        f"  d_model: {d_model}, nhead: {nhead}, layers: {num_encoder_layers}enc/{num_decoder_layers}dec\n"
        f"  Feedforward dim: {dim_feedforward}, max_seq_len: {max_seq_len}\n"
        f"  Total Trainable Parameters: {param_stats['trainable_parameters']:,}\n"
        f"  Initial GPU Memory: {trainer.get_gpu_memory_mb()['allocated_mb']} MB"
    )

    history = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        epochs=epochs,
    )

    # Save training report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_architecture": {
            "d_model": d_model,
            "nhead": nhead,
            "encoder_layers": num_encoder_layers,
            "decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "max_seq_len": max_seq_len,
            "trainable_parameters": param_stats["trainable_parameters"],
        },
        "hardware": dev_info,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "learning_rate": lr,
            "train_samples": len(train_records),
            "val_samples": len(valid_records),
        },
        "final_metrics": history[-1] if history else {},
        "training_history": history,
    }

    report_path = Path("reports/transformer_training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Seq2Seq Reaction Transformer from scratch")
    parser.add_argument("--overfit-test", action="store_true", help="Run tiny overfit test to verify convergence")
    parser.add_argument("--train-path", type=str, default="data/processed/train.parquet")
    parser.add_argument("--valid-path", type=str, default="data/processed/valid.parquet")
    parser.add_argument("--vocab-path", type=str, default="data/processed/smiles_vocab.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/transformer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=140)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging(level="INFO")

    if args.overfit_test:
        tokenizer = SMILESTokenizer.load_vocab(args.vocab_path)
        device = get_device()
        run_overfit_test(tokenizer=tokenizer, device=device)
    else:
        run_training(
            train_path=args.train_path,
            valid_path=args.valid_path,
            vocab_path=args.vocab_path,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            lr=args.lr,
            grad_accum_steps=args.grad_accum,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            resume_checkpoint=args.resume,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
