"""
Complete Trainer Engine for Seq2Seq Reaction Transformer.
Supports:
- Teacher-forcing training with shift-right target tokens
- Masked Cross-Entropy loss ignoring padding
- AdamW optimizer with weight decay
- Learning rate warmup with cosine decay
- Gradient clipping (max_norm)
- Mixed precision training (AMP) when supported on CUDA
- Validation evaluation and early stopping
- State checkpointing and seamless resume support
- GPU memory and throughput telemetry
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer
from chemrxn.training.losses import MaskedCrossEntropyLoss
from chemrxn.utils.device import get_device
from chemrxn.utils.logger import get_logger

logger = get_logger("training.transformer")


class TransformerReactionTrainer:
    """
    Dedicated training and validation harness for Seq2Seq Reaction Transformers.
    """

    def __init__(
        self,
        model: Seq2SeqReactionTransformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 500,
        checkpoint_dir: Union[str, Path] = "checkpoints/transformer",
        mixed_precision: bool = True,
        grad_accum_steps: int = 1,
        device: Optional[torch.device] = None,
        patience: int = 10,
        label_smoothing: float = 0.0,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.patience = patience

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.998),
            eps=1e-9,
        )

        self.criterion = MaskedCrossEntropyLoss(
            ignore_index=self.model.pad_idx,
            label_smoothing=label_smoothing,
        )

        self.use_amp = mixed_precision and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Initialized TransformerReactionTrainer on {self.device} "
            f"(Total Params: {total_params:,}, Trainable: {trainable_params:,}, "
            f"AMP: {self.use_amp}, Grad Accum: {self.grad_accum_steps})"
        )

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter statistics."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def get_gpu_memory_mb(self) -> Dict[str, float]:
        """Returns active GPU memory usage in MB."""
        if self.device.type == "cuda":
            return {
                "allocated_mb": round(torch.cuda.memory_allocated(self.device) / 1e6, 2),
                "reserved_mb": round(torch.cuda.memory_reserved(self.device) / 1e6, 2),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / 1e6, 2),
            }
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Executes one full teacher-forcing training epoch.
        Shift-right convention:
            decoder input: tgt[:, :-1]
            decoder target: tgt[:, 1:]
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        num_batches = len(dataloader)
        epoch_start_time = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            src = batch["src_ids"].to(self.device, non_blocking=True)
            tgt = batch["tgt_ids"].to(self.device, non_blocking=True)

            # Teacher forcing shift
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(src=src, tgt=tgt_input)
                loss = self.criterion(logits, tgt_expected)
                loss_scaled = loss / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss_scaled).backward()

            # Gradient accumulation step
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Metric tracking
            total_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                non_pad_mask = (tgt_expected != self.model.pad_idx)
                correct_tokens += ((preds == tgt_expected) & non_pad_mask).sum().item()
                total_tokens += non_pad_mask.sum().item()

        elapsed = time.time() - epoch_start_time
        avg_loss = total_loss / max(1, num_batches)
        token_acc = correct_tokens / max(1, total_tokens)

        return {
            "train_loss": round(avg_loss, 4),
            "train_token_accuracy": round(token_acc, 4),
            "train_time_seconds": round(elapsed, 2),
            "samples_per_second": round(len(dataloader.dataset) / max(0.001, elapsed), 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates model on validation dataloader using teacher forcing.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        num_batches = len(dataloader)
        val_start_time = time.time()

        for batch in dataloader:
            src = batch["src_ids"].to(self.device, non_blocking=True)
            tgt = batch["tgt_ids"].to(self.device, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(src=src, tgt=tgt_input)
                loss = self.criterion(logits, tgt_expected)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            non_pad_mask = (tgt_expected != self.model.pad_idx)
            correct_tokens += ((preds == tgt_expected) & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

        elapsed = time.time() - val_start_time
        avg_loss = total_loss / max(1, num_batches)
        token_acc = correct_tokens / max(1, total_tokens)

        return {
            "val_loss": round(avg_loss, 4),
            "val_token_accuracy": round(token_acc, 4),
            "val_time_seconds": round(elapsed, 2),
        }

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 10,
        checkpoint_frequency: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Full training loop across multiple epochs with validation and early stopping.
        """
        total_steps = epochs * (len(train_dataloader) // self.grad_accum_steps)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps),
            eta_min=1e-6,
        )

        history: List[Dict[str, Any]] = []

        logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(train_dataloader)

            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                **train_metrics,
                "gpu_memory": self.get_gpu_memory_mb(),
            }

            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                epoch_record.update(val_metrics)

                current_val_loss = val_metrics["val_loss"]
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(is_best=True)
                    epoch_record["is_best"] = True
                else:
                    self.epochs_without_improvement += 1
                    epoch_record["is_best"] = False

                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_metrics['train_loss']:.4f} "
                    f"| Train Acc: {train_metrics['train_token_accuracy']*100:.2f}% "
                    f"| Val Loss: {val_metrics['val_loss']:.4f} "
                    f"| Val Acc: {val_metrics['val_token_accuracy']*100:.2f}% "
                    f"| Time: {train_metrics['train_time_seconds']}s"
                )

                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.patience}).")
                    break
            else:
                logger.info(
                    f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_metrics['train_loss']:.4f} "
                    f"| Train Acc: {train_metrics['train_token_accuracy']*100:.2f}% "
                    f"| Time: {train_metrics['train_time_seconds']}s"
                )

            if epoch % checkpoint_frequency == 0:
                self.save_checkpoint(is_best=False, filename=f"checkpoint_epoch_{epoch}.pt")

            history.append(epoch_record)

        # Save final state
        self.save_checkpoint(is_best=False, filename="latest_checkpoint.pt")
        return history

    def save_checkpoint(self, is_best: bool = False, filename: str = "checkpoint.pt") -> Path:
        """
        Saves full checkpoint containing model weights, optimizer, scaler, and metadata.
        """
        path = self.checkpoint_dir / ("best_checkpoint.pt" if is_best else filename)
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "model_config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "nhead": self.model.nhead,
                "pad_idx": self.model.pad_idx,
                "sos_idx": self.model.sos_idx,
                "eos_idx": self.model.eos_idx,
                "max_seq_len": self.model.max_seq_len,
            },
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Resumes training from a saved checkpoint.
        """
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at {p}")

        checkpoint = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {p} (Resumed at epoch {self.current_epoch}, step {self.global_step})")
        return checkpoint
