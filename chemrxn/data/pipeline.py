"""
Complete Reaction Dataset Preparation Pipeline for ChemRxn-Intelligence.
Orchestrates ingestion, RDKit chemical validation, deduplication, scaffold-aware splitting,
split leakage auditing, train-only vocabulary generation, and structured serialization.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chemrxn.data.cleaner import ReactionCleaner
from chemrxn.data.ingestion import DatasetIngestion
from chemrxn.data.schemas import (
    ColumnMapping,
    DatasetStatistics,
    PipelineConfig,
    ProcessedReaction,
    SplitManifest,
)
from chemrxn.data.splitter import audit_split_leakage, deterministic_random_split, scaffold_split
from chemrxn.data.tokenizer import SMILESTokenizer
from chemrxn.utils.logger import get_logger, setup_logging

logger = get_logger("data_pipeline")


class ReactionDataPipeline:
    """
    End-to-end dataset preparation pipeline for 900,000+ organic reactions.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.cleaner = ReactionCleaner(
            keep_stereochemistry=self.config.keep_stereochemistry,
            require_single_product=self.config.require_single_product,
        )

    def run(self) -> Dict[str, Any]:
        """
        Executes the entire preparation pipeline.
        Returns a dictionary containing the manifest, statistics, and file paths.
        """
        start_time = time.time()
        setup_logging(level="INFO")
        logger.info(f"Starting ChemRxn-Intelligence Data Pipeline on '{self.config.input_path}'")
        logger.info(
            f"Config: quick_sample_limit={self.config.sample_limit}, "
            f"chunk_size={self.config.chunk_size}, split_strategy={self.config.split_strategy}, "
            f"keep_stereochemistry={self.config.keep_stereochemistry}"
        )

        out_dir = Path(self.config.output_dir)
        if not out_dir.is_absolute() and Path("../ReactionSeq2Seq_Dataset").exists():
            out_dir = (Path("..") / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stats = DatasetStatistics()
        invalid_counter: Dict[str, int] = {}
        unique_reactions: Dict[str, ProcessedReaction] = {}
        total_reagents_recorded = 0

        # 1. Ingestion & Cleaning Stream
        ingestion = DatasetIngestion(
            filepath=self.config.input_path,
            column_mapping=self.config.column_mapping,
            chunk_size=self.config.chunk_size,
            sample_limit=self.config.sample_limit,
        )

        for raw in ingestion.stream_raw_records():
            stats.total_rows_read += 1
            if stats.total_rows_read % 10000 == 0:
                logger.info(f"Processed {stats.total_rows_read:,} rows | Retained {len(unique_reactions):,} unique valid reactions")

            # Clean and validate
            processed, err = self.cleaner.clean_reaction(
                rxn_str=raw.get("raw_reaction_string"),
                reactants_list=[raw["reactants"]] if raw.get("reactants") else None,
                products_list=[raw["products"]] if raw.get("products") else None,
                reagents_list=[raw["reagents"]] if raw.get("reagents") else None,
                reaction_id=raw.get("reaction_id", ""),
                conditions=raw.get("conditions", {}),
            )

            if err or processed is None:
                err_category = err.split(":")[0] if err else "unknown_error"
                invalid_counter[err_category] = invalid_counter.get(err_category, 0) + 1
                continue

            # Deduplication key: canonical reactants and products
            dedup_key = f"{processed.reactants_smiles}>>{processed.products_smiles}"
            if dedup_key in unique_reactions:
                stats.duplicate_records_removed += 1
                # If existing entry has no reagents or conditions, update with new info if present
                existing = unique_reactions[dedup_key]
                if not existing.reagents_smiles and processed.reagents_smiles:
                    existing.reagents_smiles = processed.reagents_smiles
                    existing.reaction_smiles = f"{existing.reactants_smiles}>{existing.reagents_smiles}>{existing.products_smiles}"
                if not existing.conditions and processed.conditions:
                    existing.conditions = processed.conditions
            else:
                unique_reactions[dedup_key] = processed
                if processed.reagents_smiles:
                    total_reagents_recorded += 1

        retained_records = list(unique_reactions.values())
        stats.retained_valid_records = len(retained_records)
        stats.invalid_breakdown = invalid_counter
        stats.reagents_recorded_count = total_reagents_recorded

        logger.info(
            f"Ingestion & Cleaning Completed: {stats.total_rows_read:,} read, "
            f"{stats.retained_valid_records:,} retained, {stats.duplicate_records_removed:,} duplicates removed, "
            f"{sum(invalid_counter.values()):,} invalid."
        )

        if not retained_records:
            raise ValueError("No valid reaction records were retained from the dataset!")

        # 2. Deterministic Splitting
        if self.config.split_strategy == "scaffold":
            train_recs, val_recs, test_recs = scaffold_split(
                retained_records,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                seed=self.config.seed,
            )
        else:
            train_recs, val_recs, test_recs = deterministic_random_split(
                retained_records,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                seed=self.config.seed,
            )

        total_split = len(train_recs) + len(val_recs) + len(test_recs)
        stats.split_counts = {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
            "total": total_split,
        }
        stats.split_ratios_actual = {
            "train": round(len(train_recs) / total_split, 4),
            "val": round(len(val_recs) / total_split, 4),
            "test": round(len(test_recs) / total_split, 4),
        }

        # 3. Leakage Audit
        logger.info("Auditing data splits for duplicate leakage and scaffold overlaps...")
        leakage_report = audit_split_leakage(train_recs, val_recs, test_recs)
        stats.split_overlap = leakage_report
        logger.info(f"Split leakage audit: {leakage_report['exact_reaction_leakage']}")

        # 4. Tokenizer & Vocabulary Construction from TRAINING DATA ONLY
        logger.info("Building SMILES vocabulary strictly from training partition...")
        tokenizer = SMILESTokenizer()
        train_smiles_stream = []
        for r in train_recs:
            train_smiles_stream.append(r.reactants_smiles)
            train_smiles_stream.append(r.products_smiles)
            if r.reagents_smiles:
                train_smiles_stream.append(r.reagents_smiles)

        tokenizer.build_vocab(train_smiles_stream, min_frequency=self.config.vocab_min_frequency)
        vocab_path = out_dir / "smiles_vocab.json"
        tokenizer.save_vocab(vocab_path)
        stats.vocab_size = tokenizer.vocab_size
        stats.special_tokens = tokenizer.SPECIAL_TOKENS
        logger.info(f"Vocabulary constructed: {tokenizer.vocab_size} tokens saved to '{vocab_path}'")

        # 5. Serialization (Parquet & CSV)
        train_files = self._save_partition(train_recs, out_dir / "train", self.config.output_format)
        val_files = self._save_partition(val_recs, out_dir / "valid", self.config.output_format)
        test_files = self._save_partition(test_recs, out_dir / "test", self.config.output_format)

        stats.execution_time_seconds = round(time.time() - start_time, 2)

        # 6. Save Manifest & Metadata
        manifest = SplitManifest(
            created_at=datetime.datetime.now().isoformat(),
            seed=self.config.seed,
            split_strategy=self.config.split_strategy,
            keep_stereochemistry=self.config.keep_stereochemistry,
            train_file=str(train_files[0]),
            valid_file=str(val_files[0]),
            test_file=str(test_files[0]),
            vocab_file=str(vocab_path),
            statistics=stats,
        )

        manifest_path = out_dir / "split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        stats_path = out_dir / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(stats.model_dump_json(indent=2))

        config_path = out_dir / "pipeline_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self.config.model_dump_json(indent=2))

        logger.info(f"Pipeline successfully completed in {stats.execution_time_seconds}s!")
        return {
            "manifest_path": str(manifest_path),
            "stats_path": str(stats_path),
            "vocab_path": str(vocab_path),
            "config_path": str(config_path),
            "stats": stats.model_dump(),
        }

    def _save_partition(
        self, records: List[ProcessedReaction], base_path: Path, output_format: str
    ) -> List[Path]:
        """Saves a partition of records to Parquet and/or CSV."""
        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)
        saved_paths = []

        if output_format in ("parquet", "both"):
            p_path = base_path.with_suffix(".parquet")
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(p_path), compression="snappy")
            saved_paths.append(p_path)

        if output_format in ("csv", "both"):
            c_path = base_path.with_suffix(".csv")
            df.to_csv(c_path, index=False)
            saved_paths.append(c_path)

        return saved_paths


def main():
    parser = argparse.ArgumentParser(description="ChemRxn-Intelligence Reaction Data Pipeline")
    parser.add_argument("--input", type=str, default="ReactionSeq2Seq_Dataset/US_patents_1976-Sep2016_1product_reactions_valid.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--format", type=str, choices=["parquet", "csv", "both"], default="both")
    parser.add_argument("--split-strategy", type=str, choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--sample-limit", type=int, default=None, help="Max rows to process (null for full)")
    parser.add_argument("--quick-test", action="store_true", help="Quick-test mode (processes 2,000 rows)")
    parser.add_argument("--no-stereochemistry", action="store_true", help="Strip stereochemistry")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    sample_limit = 2000 if args.quick_test else args.sample_limit

    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        output_format=args.format,
        chunk_size=args.chunk_size,
        sample_limit=sample_limit,
        keep_stereochemistry=not args.no_stereochemistry,
        split_strategy=args.split_strategy,
        seed=args.seed,
    )

    pipeline = ReactionDataPipeline(config)
    results = pipeline.run()

    print("\n" + "=" * 65)
    print("  CHEMRXN-INTELLIGENCE: DATASET PIPELINE COMPLETED")
    print("=" * 65)
    s = results["stats"]
    print(f"Total Rows Read       : {s['total_rows_read']:,}")
    print(f"Retained Valid Rows   : {s['retained_valid_records']:,}")
    print(f"Duplicates Removed    : {s['duplicate_records_removed']:,}")
    print(f"Invalid Rows          : {sum(s['invalid_breakdown'].values()):,}")
    print(f"Split Counts          : Train={s['split_counts']['train']}, Val={s['split_counts']['val']}, Test={s['split_counts']['test']}")
    print(f"Vocabulary Size       : {s['vocab_size']} tokens (Built from TRAIN SET only)")
    print(f"Split Overlap Check   : {s['split_overlap']['exact_reaction_leakage']}")
    print(f"Manifest File         : {results['manifest_path']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
