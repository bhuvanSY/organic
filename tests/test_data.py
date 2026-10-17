"""
Unit tests for SMILES tokenizer, dataset loading, and batch collations.
"""

from pathlib import Path
import pytest
import torch

from chemrxn.data.dataloader import create_reaction_dataloader, reaction_collate_fn
from chemrxn.data.dataset import ReactionDataset, ReactionRecord, load_reaction_samples
from chemrxn.data.tokenizer import SMILESTokenizer


def test_tokenizer_encoding_decoding():
    tokenizer = SMILESTokenizer()
    corpus = ["CC(=O)O", "CCN", "c1ccccc1", "ClC(=O)C"]
    tokenizer.build_vocab(corpus)

    test_smiles = "CC(=O)O"
    encoded = tokenizer.encode(test_smiles, add_special_tokens=True)
    assert encoded[0] == tokenizer.sos_id
    assert encoded[-1] == tokenizer.eos_id

    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == test_smiles


def test_dataset_and_collation():
    records = [
        ReactionRecord(reactants="CC(=O)Cl.CCN", products="CC(=O)NCC"),
        ReactionRecord(reactants="c1ccccc1.Cl", products="Clc1ccccc1"),
    ]
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab([r.reactants for r in records] + [r.products for r in records])

    dataset = ReactionDataset(
        records=records,
        tokenizer=tokenizer,
        max_seq_length=32,
        compute_fingerprints=True,
    )
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "src_ids" in item0
    assert "tgt_ids" in item0
    assert "src_fp" in item0
    assert item0["src_ids"].shape[0] == 32

    loader = create_reaction_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["batch_size"] == 2
    assert batch["src_ids"].shape == (2, 32)
    assert batch["src_fp"].shape == (2, 2048)


def test_sample_loading_from_file():
    valid_file = Path("ReactionSeq2Seq_Dataset/US_patents_1976-Sep2016_1product_reactions_valid.csv")
    if valid_file.exists():
        records = load_reaction_samples(valid_file, max_samples=5)
        assert len(records) == 5
        for r in records:
            assert len(r.reactants) > 0
            assert len(r.products) > 0
