"""
Memory-efficient Reaction Dataset loader for ChemRxn-Intelligence.
Optimized for 900,000+ reaction datasets on laptop systems with RTX 3050/4050.
Supports chunked reading, line indexing, and PyTorch Dataset integration.
"""

import csv
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from chemrxn.chemistry.fingerprints import compute_morgan_fingerprint
from chemrxn.chemistry.reactions import parse_reaction_smiles
from chemrxn.chemistry.smiles import canonicalize_smiles, strip_atom_mapping
from chemrxn.data.tokenizer import SMILESTokenizer


class ReactionRecord:
    """Represents a single parsed reaction sample."""
    __slots__ = ("reactants", "products", "reagents", "reaction_smiles", "id")

    def __init__(
        self,
        reactants: str,
        products: str,
        reagents: str = "",
        reaction_smiles: str = "",
        id: Optional[str] = None,
    ):
        self.reactants = reactants
        self.products = products
        self.reagents = reagents
        self.reaction_smiles = reaction_smiles
        self.id = id

    def __repr__(self) -> str:
        return f"<ReactionRecord reactants='{self.reactants[:30]}...' products='{self.products[:30]}...'>"


def parse_uspto_line(line: str) -> Optional[ReactionRecord]:
    """
    Parses a line from Daniel Lowe's US patents TSV/CSV format.
    Columns: Source, Target, CanonicalizedReaction, OriginalReaction, PatentNumber, ParagraphNum, Year, TextMinedYield, CalculatedYield
    """
    if not line or line.startswith("#"):
        return None
    
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None
    
    # If header line
    if parts[0] == "Source" and parts[1] == "Target":
        return None

    canonical_rxn = parts[2] if len(parts) > 2 else ""
    if ">" in canonical_rxn:
        parsed = parse_reaction_smiles(canonical_rxn)
        reactants = ".".join(parsed["reactants"])
        reagents = ".".join(parsed["reagents"])
        products = ".".join(parsed["products"])
    else:
        # Fallback to Source and Target columns
        # In this dataset, Source is tokenized with spaces (e.g. 'C S ( = O ) ...')
        src_raw = parts[0].replace(" ", "")
        tgt_raw = parts[1].replace(" ", "")
        if ">" in src_raw:
            src_parts = src_raw.split(">")
            reactants = src_parts[0]
            reagents = src_parts[1] if len(src_parts) > 1 else ""
        else:
            reactants = src_raw
            reagents = ""
        products = tgt_raw

    # Clean atom mapping and canonicalize
    reactants_clean = strip_atom_mapping(reactants)
    products_clean = strip_atom_mapping(products)

    if not reactants_clean or not products_clean:
        return None

    return ReactionRecord(
        reactants=reactants_clean,
        products=products_clean,
        reagents=reagents,
        reaction_smiles=f"{reactants_clean}>{reagents}>{products_clean}",
        id=parts[4] if len(parts) > 4 else None,
    )


def parse_jin_line(line: str) -> Optional[ReactionRecord]:
    """
    Parses a line from Jin's USPTO dataset format:
    reactants>>product bond_changes
    """
    if not line or line.startswith("#"):
        return None
    
    cleaned = line.strip()
    if " " in cleaned:
        cleaned = cleaned.split(" ")[0].strip()
    
    if ">>" not in cleaned:
        return None
        
    parts = cleaned.split(">>")
    reactants_raw, product_raw = parts[0], parts[1]
    
    reactants_clean = strip_atom_mapping(reactants_raw)
    products_clean = strip_atom_mapping(product_raw)
    
    if not reactants_clean or not products_clean:
        return None

    return ReactionRecord(
        reactants=reactants_clean,
        products=products_clean,
        reagents="",
        reaction_smiles=f"{reactants_clean}>>{products_clean}",
    )


def load_reaction_samples(
    filepath: Union[str, Path],
    max_samples: Optional[int] = None,
    chunk_size: int = 10000,
) -> List[ReactionRecord]:
    """
    Loads reaction samples from either TSV/CSV or Jin format text file.
    Uses streaming iteration to keep memory bounded.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    records: List[ReactionRecord] = []
    is_jin = p.suffix == ".txt" or "Jin" in p.name
    parser_fn = parse_jin_line if is_jin else parse_uspto_line

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = parser_fn(line)
            if record is not None:
                records.append(record)
                if max_samples is not None and len(records) >= max_samples:
                    break

    return records


class ReactionDataset(Dataset):
    """
    PyTorch Dataset for reaction prediction.
    Supports either preloaded in-memory records (for rapid development/testing)
    or tokenized / fingerprint representations.
    """

    def __init__(
        self,
        records: List[ReactionRecord],
        tokenizer: Optional[SMILESTokenizer] = None,
        max_seq_length: int = 140,
        compute_fingerprints: bool = False,
        fp_nbits: int = 2048,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.compute_fingerprints = compute_fingerprints
        self.fp_nbits = fp_nbits

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        data: Dict[str, Any] = {
            "reactants_smiles": item.reactants,
            "products_smiles": item.products,
        }

        if self.tokenizer is not None:
            src_ids = self.tokenizer.encode(
                item.reactants,
                add_special_tokens=True,
                max_length=self.max_seq_length,
            )
            tgt_ids = self.tokenizer.encode(
                item.products,
                add_special_tokens=True,
                max_length=self.max_seq_length,
            )
            data["src_ids"] = torch.tensor(src_ids, dtype=torch.long)
            data["tgt_ids"] = torch.tensor(tgt_ids, dtype=torch.long)

        if self.compute_fingerprints:
            src_fp = compute_morgan_fingerprint(item.reactants, nbits=self.fp_nbits)
            tgt_fp = compute_morgan_fingerprint(item.products, nbits=self.fp_nbits)
            if src_fp is not None and tgt_fp is not None:
                data["src_fp"] = torch.from_numpy(src_fp)
                data["tgt_fp"] = torch.from_numpy(tgt_fp)
            else:
                data["src_fp"] = torch.zeros(self.fp_nbits, dtype=torch.float32)
                data["tgt_fp"] = torch.zeros(self.fp_nbits, dtype=torch.float32)

        return data
