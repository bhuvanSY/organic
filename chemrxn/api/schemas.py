"""
Pydantic v2 schemas for ChemRxn-Intelligence REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    project_name: str = "ChemRxn-Intelligence"
    torch_version: str
    rdkit_version: str
    cuda_available: bool
    device_name: str
    device_type: str


class HardwareResponse(BaseModel):
    cuda_available: bool
    torch_version: str
    device_type: str
    device_name: str
    total_memory_gb: Optional[Any] = None
    allocated_memory_mb: Optional[float] = None
    reserved_memory_mb: Optional[float] = None
    multi_processor_count: Optional[int] = None
    compute_capability: Optional[str] = None
    supports_fp16: Optional[bool] = None
    supports_bf16: Optional[bool] = None
    cpu_count: Optional[int] = None


class SMILESValidationRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to validate", json_schema_extra={"example": "CC(=O)Oc1ccccc1C(=O)O"})
    remove_mapping: bool = Field(True, description="Whether to strip atom mapping numbers")


class MolecularDescriptors(BaseModel):
    smiles: str
    molecular_formula: str
    molecular_weight: float
    logp: float
    tpsa: float
    num_heavy_atoms: int
    num_rotatable_bonds: int
    num_h_donors: int
    num_h_acceptors: int
    num_aromatic_rings: int
    num_saturated_rings: int


class SMILESValidationResponse(BaseModel):
    is_valid: bool
    input_smiles: str
    canonical_smiles: Optional[str] = None
    stripped_smiles: Optional[str] = None
    descriptors: Optional[MolecularDescriptors] = None
    error: Optional[str] = None


class PredictionRequest(BaseModel):
    reactant_smiles: str = Field(..., description="Reactant SMILES string (e.g. 'CC(=O)Cl.N')", json_schema_extra={"example": "CC(=O)Cl.CCN"})
    num_candidates: int = Field(3, ge=1, le=10, description="Number of candidate products to return")
    beam_width: int = Field(5, ge=1, le=20, description="Beam search width")


class CandidateProduct(BaseModel):
    rank: int
    product_smiles: str
    is_chemically_valid: bool
    confidence_score: float
    tanimoto_to_reactants: float
    transformer_score: Optional[float] = None
    gnn_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    descriptors: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    success: bool
    reactants_input: str
    reactants_canonical: Optional[str] = None
    reactant_descriptors: Optional[Dict[str, Any]] = None
    model_status: str
    is_trained: bool
    candidates: List[CandidateProduct] = []
    hardware_device: str
    error: Optional[str] = None


class DatasetFileInfo(BaseModel):
    filename: str
    size_mb: float
    format: str
    description: str


class DatasetSummaryResponse(BaseModel):
    dataset_directory: str
    total_files: int
    total_size_mb: float
    files: List[DatasetFileInfo]
    train_record_estimate: str


class ReactionClassResponse(BaseModel):
    reactants_input: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    is_generative: bool = False
    distinction_note: str = "Reaction classification predicts transformation type only; it does NOT generate product molecules."


class RetrievalCandidate(BaseModel):
    rank: int
    reaction_id: str
    retrieved_reactants: str
    retrieved_product: str
    similarity_score: float


class RetrievalResponse(BaseModel):
    query_reactants: str
    candidates: List[RetrievalCandidate]
    latency_ms: float


class CandidateRankingRequest(BaseModel):
    reactants: str
    candidates: List[str]


class RankedCandidateItem(BaseModel):
    rank: int
    candidate_smiles: str
    plausibility_score: float


class CandidateRankingResponse(BaseModel):
    reactants: str
    ranked_candidates: List[RankedCandidateItem]
    latency_ms: float


class StructureRequest(BaseModel):
    smiles: str = Field(..., description="SMILES to generate 3D coordinates for", json_schema_extra={"example": "CC(=O)Oc1ccccc1C(=O)O"})
    add_hydrogens: bool = Field(True, description="Whether to embed explicit hydrogens")
    optimize_mmff: bool = Field(True, description="Whether to minimize energy via MMFF94 force field")


class ReactionCompareRequest(BaseModel):
    reactant_smiles: str = Field(..., description="Reactant SMILES", json_schema_extra={"example": "CC(=O)Cl.c1ccccc1"})
    product_smiles: str = Field(..., description="Product SMILES", json_schema_extra={"example": "CC(=O)c1ccccc1"})


class ModelCheckpointInfo(BaseModel):
    name: str
    checkpoint_path: str
    exists: bool
    size_mb: Optional[float] = None
    architecture: str
    trained_status: str


class ModelStatusResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    models: List[ModelCheckpointInfo]
    ready_for_inference: bool

