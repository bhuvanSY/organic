"""
DataLoader and Collation utilities for ChemRxn-Intelligence.
Windows-friendly multiprocessing handling and GPU memory optimization.
"""

from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader

from chemrxn.data.dataset import ReactionDataset


def reaction_collate_fn(batch: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    """
    Collates a list of sample dictionaries into a padded batch dictionary.
    Generates attention/padding masks for transformer models.
    """
    reactants_smiles = [item["reactants_smiles"] for item in batch]
    products_smiles = [item["products_smiles"] for item in batch]

    collated: Dict[str, Any] = {
        "reactants_smiles": reactants_smiles,
        "products_smiles": products_smiles,
        "batch_size": len(batch),
    }

    if "src_ids" in batch[0] and "tgt_ids" in batch[0]:
        src_ids_list = [item["src_ids"] for item in batch]
        tgt_ids_list = [item["tgt_ids"] for item in batch]

        src_padded = torch.stack(src_ids_list, dim=0)
        tgt_padded = torch.stack(tgt_ids_list, dim=0)

        # src_key_padding_mask: True where padded (for PyTorch nn.Transformer)
        src_padding_mask = (src_padded == pad_id)
        tgt_padding_mask = (tgt_padded == pad_id)

        collated["src_ids"] = src_padded
        collated["tgt_ids"] = tgt_padded
        collated["src_padding_mask"] = src_padding_mask
        collated["tgt_padding_mask"] = tgt_padding_mask

    if "src_fp" in batch[0] and "tgt_fp" in batch[0]:
        collated["src_fp"] = torch.stack([item["src_fp"] for item in batch], dim=0)
        collated["tgt_fp"] = torch.stack([item["tgt_fp"] for item in batch], dim=0)

    return collated


def create_reaction_dataloader(
    dataset: ReactionDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    pad_id: int = 0,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader configured safely for Windows laptops.
    """
    # On Windows, num_workers > 0 can trigger pickling and memory issues with CUDA
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=lambda b: reaction_collate_fn(b, pad_id=pad_id),
    )
