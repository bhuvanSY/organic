"""
Automated test suite for ChemRxn-Intelligence Reaction Data Preparation Pipeline.
Tests chemical cleaning, stereochemistry, invalid structures, deduplication,
scaffold splitting, zero-leakage auditing, and train-only vocabulary creation.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chemrxn.data.cleaner import ReactionCleaner
from chemrxn.data.ingestion import DatasetIngestion, auto_detect_mapping
from chemrxn.data.pipeline import ReactionDataPipeline
from chemrxn.data.schemas import ColumnMapping, PipelineConfig, ProcessedReaction
from chemrxn.data.splitter import audit_split_leakage, deterministic_random_split, scaffold_split
from chemrxn.data.tokenizer import SMILESTokenizer


def test_cleaner_known_valid_reactions():
    cleaner = ReactionCleaner(keep_stereochemistry=True, require_single_product=True)
    # 1. Simple acylation
    rxn = "CC(=O)Cl.CCN>>CC(=O)NCC"
    processed, err = cleaner.clean_reaction(rxn_str=rxn)
    assert err is None
    assert processed is not None
    assert processed.reactants_smiles == "CC(=O)Cl.CCN"
    assert processed.products_smiles == "CCNC(C)=O" or "CC(=O)NCC" in processed.products_smiles

    # 2. Reaction with reagents in middle
    rxn3 = "CC(=O)O.CCO>CCN(CC)CC>CC(=O)OCC"
    processed3, err3 = cleaner.clean_reaction(rxn_str=rxn3)
    assert err3 is None
    assert processed3 is not None
    assert processed3.reagents_smiles == "CCN(CC)CC"
    assert processed3.products_smiles == "CCOC(C)=O"


def test_cleaner_stereochemistry_preservation():
    # Chiral reactant and product
    chiral_rxn = "C[C@H](O)Cl.N>>C[C@H](O)N"
    cleaner_keep = ReactionCleaner(keep_stereochemistry=True)
    proc_keep, err_keep = cleaner_keep.clean_reaction(rxn_str=chiral_rxn)
    assert err_keep is None
    assert "@" in proc_keep.products_smiles
    assert "@" in proc_keep.reactants_smiles

    # When stereochemistry is explicitly disabled
    cleaner_drop = ReactionCleaner(keep_stereochemistry=False)
    proc_drop, err_drop = cleaner_drop.clean_reaction(rxn_str=chiral_rxn)
    assert err_drop is None
    assert "@" not in proc_drop.products_smiles


def test_cleaner_atom_mapping_removal():
    mapped_rxn = "[CH3:1][C:2](=[O:3])[Cl:4].[NH2:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[NH:5][CH2:6][CH3:7]"
    cleaner = ReactionCleaner()
    proc, err = cleaner.clean_reaction(rxn_str=mapped_rxn)
    assert err is None
    assert ":" not in proc.reactants_smiles
    assert ":" not in proc.products_smiles
    assert proc.products_smiles == "CCNC(C)=O"


def test_cleaner_known_invalid_and_unsupported_smiles():
    cleaner = ReactionCleaner(require_single_product=True)

    # 1. Empty reaction string
    proc, err = cleaner.clean_reaction(rxn_str="")
    assert err == "empty_input"

    # 2. Missing '>' delimiter
    proc, err = cleaner.clean_reaction(rxn_str="CCO.CC(=O)Cl")
    assert "malformed_syntax" in err

    # 3. Invalid reactant chemistry
    proc, err = cleaner.clean_reaction(rxn_str="NotAValidChemicalReactant!!!>>CCO")
    assert "invalid_reactant" in err

    # 4. Invalid product chemistry
    proc, err = cleaner.clean_reaction(rxn_str="CCO>>NotAValidProductMolecule???")
    assert "invalid_product" in err

    # 5. Empty product
    proc, err = cleaner.clean_reaction(rxn_str="CCO>>")
    assert "empty_products" in err

    # 6. Multi-product rejection (when single product required)
    proc, err = cleaner.clean_reaction(rxn_str="CC(=O)Cl.CCN>>CC(=O)NCC.Cl")
    assert "multi_product" in err

    # 7. Hypervalent invalid nitrogen / impossible structure
    proc, err = cleaner.clean_reaction(rxn_str="CN(C)(C)(C)(C)C>>CCO")
    assert "invalid_reactant" in err


def test_scaffold_extraction():
    cleaner = ReactionCleaner()
    # Aromatic ring product has scaffold
    scaff_benzene = cleaner.extract_scaffold("c1ccccc1CC(=O)O")
    assert scaff_benzene == "c1ccccc1"

    # Acyclic product returns '_acyclic_'
    scaff_acyclic = cleaner.extract_scaffold("CCCC(=O)O")
    assert scaff_acyclic == "_acyclic_"


def test_scaffold_split_and_leakage_audit():
    # Create sample reactions with different product scaffolds
    reactions = []
    # Scaffold 1: Benzene ring products
    for i in range(20):
        reactions.append(
            ProcessedReaction(
                reaction_id=f"rxn_b_{i}",
                reactants_smiles=f"c1ccccc1.C{i}Cl",
                products_smiles=f"c1ccccc1C{i}",
                product_scaffold="c1ccccc1",
            )
        )
    # Scaffold 2: Pyridine ring products
    for i in range(10):
        reactions.append(
            ProcessedReaction(
                reaction_id=f"rxn_p_{i}",
                reactants_smiles=f"c1ccncc1.C{i}Cl",
                products_smiles=f"c1ccncc1C{i}",
                product_scaffold="c1ccncc1",
            )
        )
    # Scaffold 3: Acyclic products
    for i in range(10):
        reactions.append(
            ProcessedReaction(
                reaction_id=f"rxn_a_{i}",
                reactants_smiles=f"CC(=O)O.C{i}O",
                products_smiles=f"CC(=O)OC{i}",
                product_scaffold="_acyclic_",
            )
        )

    train, val, test = scaffold_split(reactions, train_ratio=0.75, val_ratio=0.125, test_ratio=0.125, seed=42)
    assert len(train) + len(val) + len(test) == 40
    assert len(train) > len(val)
    assert len(train) > len(test)

    # Audit leakage
    audit = audit_split_leakage(train, val, test)
    assert audit["exact_reaction_leakage"]["total_reaction_leakage"] == 0
    assert audit["exact_reaction_leakage"]["train_test_overlap_count"] == 0
    assert audit["product_molecule_leakage"]["train_test_product_overlap_count"] == 0


def test_leakage_detector_catches_injected_duplicate():
    r1 = ProcessedReaction(reaction_id="1", reactants_smiles="CC.Cl", products_smiles="CCCl")
    r2 = ProcessedReaction(reaction_id="2", reactants_smiles="CC.Br", products_smiles="CCBr")
    # Deliberate duplicate across train and test
    train = [r1]
    test = [r1, r2]
    val = [r2]

    audit = audit_split_leakage(train, val, test)
    assert audit["exact_reaction_leakage"]["train_test_overlap_count"] == 1
    assert audit["exact_reaction_leakage"]["total_reaction_leakage"] > 0


def test_train_only_vocabulary_isolation():
    train_rxns = [
        ProcessedReaction(reaction_id="1", reactants_smiles="CC(=O)Cl.CCN", products_smiles="CC(=O)NCC"),
    ]
    test_rxns = [
        # Contains Boron 'B' and Fluorine 'F', which are NOT in train
        ProcessedReaction(reaction_id="2", reactants_smiles="c1ccccc1B(O)O.F", products_smiles="Fc1ccccc1"),
    ]

    tokenizer = SMILESTokenizer()
    # Build vocabulary strictly from train
    tokenizer.build_vocab([r.reactants_smiles for r in train_rxns] + [r.products_smiles for r in train_rxns])

    # Boron 'B' should NOT be in train vocabulary
    assert "B" not in tokenizer.token_to_id

    # Encoding test string containing unseen 'B' should map to <unk>
    encoded = tokenizer.encode("B(O)O", add_special_tokens=False)
    assert tokenizer.unk_id in encoded


def test_end_to_end_pipeline_parquet_and_manifest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a small CSV file
        raw_csv = Path(tmp_dir) / "test_reactions.csv"
        df = pd.DataFrame({
            "reaction_smiles": [
                "CC(=O)Cl.CCN>>CC(=O)NCC",
                "c1ccccc1.Cl>>Clc1ccccc1",
                "CC(=O)O.CCO>>CCOC(C)=O",
                "CC(=O)O.CCO>>CCOC(C)=O",  # duplicate
                "InvalidChem>>CCO",         # invalid
            ],
            "id": ["rxn1", "rxn2", "rxn3", "rxn4", "rxn5"],
        })
        df.to_csv(raw_csv, index=False)

        config = PipelineConfig(
            input_path=str(raw_csv),
            output_dir=str(Path(tmp_dir) / "output"),
            output_format="both",
            chunk_size=10,
            sample_limit=None,
            split_strategy="random",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        pipeline = ReactionDataPipeline(config)
        results = pipeline.run()

        stats = results["stats"]
        assert stats["total_rows_read"] == 5
        assert stats["retained_valid_records"] == 3
        assert stats["duplicate_records_removed"] == 1
        assert "invalid_reactant" in stats["invalid_breakdown"]
        assert stats["split_overlap"]["exact_reaction_leakage"]["total_reaction_leakage"] == 0

        # Check files exist
        out_path = Path(tmp_dir) / "output"
        assert (out_path / "train.parquet").exists()
        assert (out_path / "train.csv").exists()
        assert (out_path / "valid.parquet").exists()
        assert (out_path / "test.parquet").exists()
        assert (out_path / "split_manifest.json").exists()
        assert (out_path / "dataset_stats.json").exists()
        assert (out_path / "smiles_vocab.json").exists()
