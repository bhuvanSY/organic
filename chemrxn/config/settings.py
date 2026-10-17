"""
Configuration settings for ChemRxn-Intelligence using Pydantic.
Supports YAML file loading, environment overrides, and hardware auto-adaptation.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    project_name: str = "ChemRxn-Intelligence"
    version: str = "0.1.0"
    seed: int = 42
    deterministic: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/chemrxn.log"


class HardwareConfig(BaseModel):
    device: str = "auto"
    cuda_device_id: int = 0
    mixed_precision: bool = True
    cudnn_benchmark: bool = False
    pin_memory: bool = True
    num_workers: int = 0
    empty_cache_frequency: int = 100


class DataConfig(BaseModel):
    raw_dir: str = "ReactionSeq2Seq_Dataset"
    processed_dir: str = "data/processed"
    train_file: str = "ReactionSeq2Seq_Dataset/US_patents_1976-Sep2016_1product_reactions_train.csv"
    valid_file: str = "ReactionSeq2Seq_Dataset/US_patents_1976-Sep2016_1product_reactions_valid.csv"
    test_file: str = "ReactionSeq2Seq_Dataset/US_patents_1976-Sep2016_1product_reactions_test.csv"
    jin_train_file: str = "ReactionSeq2Seq_Dataset/Jin_USPTO_1product_train.txt"
    jin_valid_file: str = "ReactionSeq2Seq_Dataset/Jin_USPTO_1product_valid.txt"
    jin_test_file: str = "ReactionSeq2Seq_Dataset/Jin_USPTO_1product_test.txt"
    max_sequence_length: int = 140
    vocab_file: str = "data/processed/smiles_vocab.json"
    sample_size_limit: Optional[int] = None
    chunk_size: int = 10000


class ChemistryConfig(BaseModel):
    fingerprint_type: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_nbits: int = 2048
    canonicalize: bool = True
    remove_atom_mapping: bool = True


class ModelConfig(BaseModel):
    type: str = "transformer"
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    init_method: str = "xavier_uniform"


class TrainingConfig(BaseModel):
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    max_epochs: int = 15
    warmup_steps: int = 500
    patience: int = 3
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 2


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8001
    reload: bool = False


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    chemistry: ChemistryConfig = Field(default_factory=ChemistryConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Loads AppConfig from YAML file if provided and exists; otherwise falls back to defaults.
    """
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return AppConfig(**raw_data)
    
    # Try looking for rtx_laptop_config.yaml or default_config.yaml
    default_paths = [
        "configs/rtx_laptop_config.yaml",
        "configs/default_config.yaml",
    ]
    for p in default_paths:
        if Path(p).exists():
            with open(p, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
            return AppConfig(**raw_data)

    return AppConfig()
