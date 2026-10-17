"""
Modular Training Engine for ChemRxn-Intelligence.
Supports CUDA mixed precision (AMP), gradient accumulation, and deterministic training.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chemrxn.utils.device import get_device, to_device
from chemrxn.utils.logger import get_logger

logger = get_logger("training")


class ReactionTrainer:
    """
    Harness for training reaction prediction neural networks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        logger.info(
            f"Initialized ReactionTrainer on {self.device} (AMP: {self.use_amp}, "
            f"Grad Accum: {self.grad_accum_steps})"
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int = 1) -> Dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = to_device(batch, self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if "src_ids" in batch and "tgt_ids" in batch:
                    # Seq2Seq Transformer input
                    src = batch["src_ids"]
                    tgt = batch["tgt_ids"]
                    # Teacher forcing: input is tgt[:, :-1], target is tgt[:, 1:]
                    tgt_input = tgt[:, :-1]
                    tgt_target = tgt[:, 1:]
                    src_mask = batch.get("src_padding_mask")
                    tgt_mask = batch.get("tgt_padding_mask")
                    if tgt_mask is not None:
                        tgt_mask = tgt_mask[:, :-1]

                    logits = self.model(
                        src=src,
                        tgt=tgt_input,
                        src_key_padding_mask=src_mask,
                        tgt_key_padding_mask=tgt_mask,
                    )
                    loss = self.criterion(logits, tgt_target)
                elif "src_fp" in batch and "tgt_fp" in batch:
                    # Fingerprint MLP input
                    preds = self.model(batch["src_fp"])
                    loss = self.criterion(preds, batch["tgt_fp"])
                else:
                    raise ValueError("Unsupported batch format for training step.")

                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps

        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch} finished. Average Train Loss: {avg_loss:.5f}")
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model loss on a validation dataloader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            batch = to_device(batch, self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if "src_ids" in batch and "tgt_ids" in batch:
                    src = batch["src_ids"]
                    tgt = batch["tgt_ids"]
                    tgt_input = tgt[:, :-1]
                    tgt_target = tgt[:, 1:]
                    src_mask = batch.get("src_padding_mask")
                    tgt_mask = batch.get("tgt_padding_mask")
                    if tgt_mask is not None:
                        tgt_mask = tgt_mask[:, :-1]

                    logits = self.model(
                        src=src,
                        tgt=tgt_input,
                        src_key_padding_mask=src_mask,
                        tgt_key_padding_mask=tgt_mask,
                    )
                    loss = self.criterion(logits, tgt_target)
                elif "src_fp" in batch and "tgt_fp" in batch:
                    preds = self.model(batch["src_fp"])
                    loss = self.criterion(preds, batch["tgt_fp"])
                else:
                    raise ValueError("Unsupported batch format for evaluation step.")

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        return {"val_loss": avg_loss}
