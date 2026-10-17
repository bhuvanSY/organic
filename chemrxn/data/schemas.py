"""
Data schemas and configuration models for reaction dataset preparation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


@dataclass
class ProcessedReaction:
    """Represents a validated, canonicalized, deduplicated reaction record."""
    reaction_id: str
    reactants_smiles: str
    products_smiles: str
    reagents_smiles: str = ""
    reaction_smiles: str = ""
    product_scaffold: str = ""
    is_valid: bool = True
    error_reason: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "reaction_id": self.reaction_id,
            "reactants_smiles": self.reactants_smiles,
            "products_smiles": self.products_smiles,
            "reagents_smiles": self.reagents_smiles,
            "reaction_smiles": self.reaction_smiles,
            "product_scaffold": self.product_scaffold,
        }
        for k, v in self.conditions.items():
            d[f"cond_{k}"] = v
        return d


class ColumnMapping(BaseModel):
    """Configurable column mapping for dataset ingestion."""
    reaction_col: Optional[str] = None
    reactants_col: Optional[str] = None
    products_col: Optional[str] = None
    reagents_col: Optional[str] = None
    id_col: Optional[str] = None
    condition_cols: List[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Pipeline execution configuration."""
    input_path: str = "ReactionSeq2Seq_Dataset/US_patents_1976-Sep2016_1product_reactions_valid.csv"
    output_dir: str = "data/processed"
    output_format: str = "both"  # "parquet", "csv", or "both"
    chunk_size: int = 5000
    sample_limit: Optional[int] = 2000  # Quick-test default; None for full 900k run
    keep_stereochemistry: bool = True
    require_single_product: bool = True
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    split_strategy: str = "scaffold"  # "scaffold" or "random"
    vocab_min_frequency: int = 1
    column_mapping: Optional[ColumnMapping] = None


class DatasetStatistics(BaseModel):
    """Comprehensive metrics and accounting for dataset preparation."""
    total_rows_read: int = 0
    retained_valid_records: int = 0
    duplicate_records_removed: int = 0
    invalid_breakdown: Dict[str, int] = Field(default_factory=dict)
    split_counts: Dict[str, int] = Field(default_factory=dict)
    split_ratios_actual: Dict[str, float] = Field(default_factory=dict)
    split_overlap: Dict[str, Any] = Field(default_factory=dict)
    scaffold_summary: Dict[str, Any] = Field(default_factory=dict)
    vocab_size: int = 0
    special_tokens: List[str] = Field(default_factory=list)
    reagents_recorded_count: int = 0
    execution_time_seconds: float = 0.0


class SplitManifest(BaseModel):
    """Manifest describing split files, records, and checksum/integrity metadata."""
    pipeline_version: str = "0.1.0"
    created_at: str
    seed: int
    split_strategy: str
    keep_stereochemistry: bool
    train_file: str
    valid_file: str
    test_file: str
    vocab_file: str
    statistics: DatasetStatistics
