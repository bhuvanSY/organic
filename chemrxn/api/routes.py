"""
API route definitions for ChemRxn-Intelligence.
"""

from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
import rdkit
import torch

from chemrxn.api.schemas import (
    CandidateProduct,
    CandidateRankingRequest,
    CandidateRankingResponse,
    DatasetFileInfo,
    DatasetSummaryResponse,
    HardwareResponse,
    HealthResponse,
    MolecularDescriptors,
    PredictionRequest,
    PredictionResponse,
    RankedCandidateItem,
    ReactionClassResponse,
    RetrievalCandidate,
    RetrievalResponse,
    SMILESValidationRequest,
    SMILESValidationResponse,
    ModelStatusResponse,
    StructureRequest,
    ReactionCompareRequest,
)
from chemrxn.chemistry.fingerprints import compute_tanimoto_similarity
from chemrxn.chemistry.smiles import (
    canonicalize_smiles,
    compute_molecular_descriptors,
    is_valid_smiles,
    strip_atom_mapping,
)
from chemrxn.inference.predictor import ReactionPredictor
from chemrxn.utils.device import get_device, get_device_info

router = APIRouter()
predictor = ReactionPredictor()


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint with runtime environment metadata."""
    device = get_device()
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    return HealthResponse(
        status="ok",
        version="0.1.0",
        project_name="ChemRxn-Intelligence",
        torch_version=torch.__version__,
        rdkit_version=rdkit.__version__,
        cuda_available=cuda_avail,
        device_name=device_name,
        device_type=device.type,
    )


@router.get("/hardware", response_model=HardwareResponse)
def get_hardware_info():
    """Detailed hardware capability inspection (GPU, VRAM, Compute capability)."""
    info = get_device_info()
    return HardwareResponse(**info)


@router.post("/validate-smiles", response_model=SMILESValidationResponse)
def validate_smiles(req: SMILESValidationRequest):
    """
    Validates a SMILES string, strips atom-mapping, canonicalizes,
    and returns molecular descriptors.
    """
    raw = req.smiles.strip()
    if not raw:
        return SMILESValidationResponse(
            is_valid=False,
            input_smiles=req.smiles,
            error="Empty SMILES string provided.",
        )

    stripped = strip_atom_mapping(raw) if req.remove_mapping else raw
    valid = is_valid_smiles(stripped)
    if not valid:
        return SMILESValidationResponse(
            is_valid=False,
            input_smiles=req.smiles,
            stripped_smiles=stripped,
            error=f"RDKit was unable to parse and sanitize the SMILES: '{raw}'",
        )

    canonical = canonicalize_smiles(stripped)
    desc_dict = compute_molecular_descriptors(canonical) if canonical else None
    descriptors = MolecularDescriptors(**desc_dict) if desc_dict else None

    return SMILESValidationResponse(
        is_valid=True,
        input_smiles=req.smiles,
        canonical_smiles=canonical,
        stripped_smiles=stripped,
        descriptors=descriptors,
    )


@router.get("/dataset-info", response_model=DatasetSummaryResponse)
def get_dataset_summary():
    """Inspects the local ReactionSeq2Seq_Dataset files and record counts."""
    dataset_dir = Path("ReactionSeq2Seq_Dataset")
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset directory not found.")

    files: List[DatasetFileInfo] = []
    total_size = 0.0

    for file_path in dataset_dir.iterdir():
        if file_path.is_file():
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            total_size += size_mb

            if "train" in file_path.name.lower():
                desc = "Training split (>900k reactions in patent CSV, ~400k in Jin)"
            elif "valid" in file_path.name.lower():
                desc = "Validation split (~50k reactions in patent CSV, ~29k in Jin)"
            elif "test" in file_path.name.lower():
                desc = "Test benchmark split (~50k reactions in patent CSV, ~38k in Jin)"
            else:
                desc = "Dataset split file"

            files.append(
                DatasetFileInfo(
                    filename=file_path.name,
                    size_mb=size_mb,
                    format="TSV/CSV" if file_path.suffix == ".csv" else "Text (Jin)",
                    description=desc,
                )
            )

    files.sort(key=lambda x: x.size_mb, reverse=True)

    return DatasetSummaryResponse(
        dataset_directory=str(dataset_dir),
        total_files=len(files),
        total_size_mb=round(total_size, 2),
        files=files,
        train_record_estimate="910,000+ patent reactions; 400,000+ Jin reactions",
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_reaction(req: PredictionRequest):
    """
    Accepts reactant SMILES and returns organic reaction prediction results.
    Strictly reports authentic model status without fabricating fake accuracy.
    """
    result = predictor.predict(
        reactant_smiles=req.reactant_smiles,
        num_candidates=req.num_candidates,
        beam_width=req.beam_width,
    )

    if not result["success"]:
        return PredictionResponse(
            success=False,
            reactants_input=req.reactant_smiles,
            model_status="error",
            is_trained=False,
            hardware_device=str(get_device()),
            error=result.get("error", "Prediction failed"),
        )

    candidates = [
        CandidateProduct(**c) for c in result.get("candidates", [])
    ]

    return PredictionResponse(
        success=True,
        reactants_input=result["reactants_input"],
        reactants_canonical=result["reactants_canonical"],
        reactant_descriptors=result["reactant_descriptors"],
        model_status=result["model_status"],
        is_trained=result["is_trained"],
        candidates=candidates,
        hardware_device=result["hardware_device"],
    )


# =========================================================================
# Classical ML Baseline Endpoints
# =========================================================================

@router.post("/baseline/classify", response_model=ReactionClassResponse)
def classify_reaction_endpoint(req: PredictionRequest):
    """
    Classifies organic transformation category from reactant SMILES.
    NOTE: This model predicts reaction type (e.g. Acylation vs Reduction);
    it does NOT generate product molecules.
    """
    from chemrxn.baselines.featurizer import ReactionFeaturizer
    from chemrxn.baselines.reaction_classifier import ReactionClassClassifier
    from chemrxn.baselines.taxonomy import ID_TO_CLASS

    clean_r = canonicalize_smiles(strip_atom_mapping(req.reactant_smiles))
    if not clean_r:
        raise HTTPException(status_code=400, detail="Invalid reactant SMILES provided.")

    ckpt_path = Path("checkpoints/baselines/reaction_classifier.joblib")
    if not ckpt_path.exists():
        raise HTTPException(status_code=503, detail="Baseline classifier checkpoint not found. Run training first.")

    classifier = ReactionClassClassifier.load(ckpt_path)
    featurizer = ReactionFeaturizer(fp_nbits=1024, include_descriptors=True)
    feat = featurizer.featurize_reactants(clean_r).reshape(1, -1)

    probs = classifier.predict_proba(feat)[0]
    top_idx = int(probs.argmax())
    prob_dict = {ID_TO_CLASS.get(i, f"Class_{i}"): round(float(p), 4) for i, p in enumerate(probs)}

    return ReactionClassResponse(
        reactants_input=req.reactant_smiles,
        predicted_class=ID_TO_CLASS.get(top_idx, "Unknown"),
        confidence=round(float(probs[top_idx]), 4),
        probabilities=prob_dict,
        is_generative=False,
    )


@router.post("/baseline/retrieve", response_model=RetrievalResponse)
def retrieve_similar_reactions(req: PredictionRequest):
    """
    Nearest-neighbor retrieval: searches historical reactions with similar reactant fingerprints.
    """
    import time
    from chemrxn.baselines.nearest_neighbor import NearestNeighborRetriever

    clean_r = canonicalize_smiles(strip_atom_mapping(req.reactant_smiles))
    if not clean_r:
        raise HTTPException(status_code=400, detail="Invalid reactant SMILES provided.")

    ckpt_path = Path("checkpoints/baselines/nearest_neighbor.joblib")
    if not ckpt_path.exists():
        raise HTTPException(status_code=503, detail="Baseline retriever checkpoint not found. Run training first.")

    retriever = NearestNeighborRetriever.load(ckpt_path)
    t0 = time.time()
    results = retriever.retrieve(clean_r, top_k=req.num_candidates)
    latency_ms = round((time.time() - t0) * 1000.0, 3)

    return RetrievalResponse(
        query_reactants=req.reactant_smiles,
        candidates=[RetrievalCandidate(**item) for item in results],
        latency_ms=latency_ms,
    )


@router.post("/baseline/rank-candidates", response_model=CandidateRankingResponse)
def rank_candidates_endpoint(req: CandidateRankingRequest):
    """
    Supervised candidate ranker: scores and orders proposed product molecules.
    """
    import time
    from chemrxn.baselines.candidate_ranker import SupervisedCandidateRanker

    clean_r = canonicalize_smiles(strip_atom_mapping(req.reactants))
    if not clean_r:
        raise HTTPException(status_code=400, detail="Invalid reactant SMILES provided.")

    ckpt_path = Path("checkpoints/baselines/candidate_ranker.joblib")
    if not ckpt_path.exists():
        raise HTTPException(status_code=503, detail="Baseline candidate ranker checkpoint not found. Run training first.")

    ranker = SupervisedCandidateRanker.load(ckpt_path)
    t0 = time.time()
    scored = ranker.score_candidates(clean_r, req.candidates)
    latency_ms = round((time.time() - t0) * 1000.0, 3)

    ranked_items = [
        RankedCandidateItem(rank=i + 1, candidate_smiles=c, plausibility_score=round(float(s), 4))
        for i, (c, s) in enumerate(scored)
    ]

    return CandidateRankingResponse(
        reactants=req.reactants,
        ranked_candidates=ranked_items,
        latency_ms=latency_ms,
    )


@router.post("/predict/reaction-centers")
def predict_reaction_centers_endpoint(req: PredictionRequest):
    """
    Predicts reacting atoms and reaction centers in reactant molecules using the GNN.
    """
    res = predictor.predict_reaction_centers(req.reactant_smiles)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res.get("error", "Failed to compute reaction centers"))
    return res


@router.get("/reports/benchmark")
def get_benchmark_report():
    """Returns the latest comparative candidate ranking benchmark report."""
    import json
    report_file = Path("reports/gnn_candidate_ranking_report.json")
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found. Run evaluation script first.")
    with open(report_file, "r") as f:
        return json.load(f)


@router.post("/predict/hybrid")
def predict_hybrid_endpoint(req: PredictionRequest):
    """
    Executes hybrid prediction combining Transformer, GNN, Baseline Ranker,
    Reaction Center Consistency, Platt Calibration, and Abstention.
    """
    from chemrxn.inference.hybrid_engine import HybridInferenceEngine
    engine = HybridInferenceEngine(lazy_load=False)
    out = engine.predict(
        reactant_smiles=req.reactant_smiles,
        beam_size=req.beam_width,
        num_candidates=req.num_candidates,
    )
    return {
        "reactant_smiles": out.reactant_smiles,
        "canonical_reactants": out.canonical_reactants,
        "abstained": out.abstained,
        "abstention_reason": out.abstention_reason,
        "top_calibrated_confidence": out.top_calibrated_confidence,
        "epistemic_uncertainty": out.epistemic_uncertainty,
        "reaction_center_atoms": out.reaction_center_atoms,
        "latency_ms": out.latency_ms,
        "candidates": [
            {
                "rank": c.rank,
                "product_smiles": c.product_smiles,
                "canonical_smiles": c.canonical_smiles,
                "is_syntactically_valid": c.is_syntactically_valid,
                "validity_note": c.validity_note,
                "calibrated_confidence": c.calibrated_confidence,
                "raw_scores": c.raw_scores,
                "molecular_descriptors": c.molecular_descriptors,
                "model_provenance": c.model_provenance,
            }
            for c in out.candidates
        ],
    }


@router.get("/reports/ablation")
def get_ablation_report():
    """Returns the systematic ablation benchmark report."""
    import json
    report_file = Path("reports/hybrid_inference_ablation_report.json")
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Ablation report not found. Run benchmark script first.")
    with open(report_file, "r") as f:
        return json.load(f)


@router.get("/render-svg")
def render_molecule_svg(smiles: str, width: int = 320, height: int = 220):
    """
    Renders a crisp 2D vector SVG depiction of a SMILES string with transparent background.
    """
    from fastapi.responses import Response
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    s = smiles.strip()
    if not s:
        raise HTTPException(status_code=400, detail="Empty SMILES provided.")

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        err_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
            <rect width="100%" height="100%" fill="#F7F7F2" rx="8" stroke="#E5E6DE" stroke-width="1"/>
            <text x="50%" y="45%" text-anchor="middle" fill="#87966A" font-family="Inter, sans-serif" font-size="12" font-weight="600">STRUCTURE UNPARSEABLE</text>
            <text x="50%" y="60%" text-anchor="middle" fill="#71716B" font-family="Inter, sans-serif" font-size="11">Invalid chemical valence / syntax</text>
        </svg>"""
        return Response(content=err_svg, media_type="image/svg+xml")

    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2.0
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        return Response(content=svg_text, media_type="image/svg+xml")
    except Exception as e:
        err_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
            <rect width="100%" height="100%" fill="#F7F7F2" rx="8"/>
            <text x="50%" y="50%" text-anchor="middle" fill="#EF4444" font-size="12">Render error: {str(e)[:30]}</text>
        </svg>"""
        return Response(content=err_svg, media_type="image/svg+xml")


@router.get("/reports/evaluation")
def get_rigorous_evaluation_report():
    """Returns the comprehensive benchmark evaluation report across all five model families."""
    import json
    report_file = Path("reports/evaluation/evaluation_results.json")
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run benchmark suite first.")
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/reports/rl")
def get_rl_policy_report():
    """Returns the experimental RL candidate selector evaluation telemetry."""
    import json
    report_file = Path("reports/rl_candidate_policy_report.json")
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="RL report not found.")
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Production Master Endpoints (Standardized REST contracts)
# =========================================================================

@router.get("/model/status", response_model=ModelStatusResponse)
def get_model_status():
    """
    Reports whether the actual model checkpoints are available and ready for inference.
    """
    from chemrxn.api.schemas import ModelCheckpointInfo
    device = get_device()
    ckpts = [
        ("Seq2Seq Reaction Transformer", "checkpoints/transformer/best_checkpoint.pt", "4-layer Encoder-Decoder, 8 Heads", "Trained on USPTO Stereo Mixed Split"),
        ("Molecular Reaction GNN (MPNN)", "checkpoints/gnn/best_gnn.pt", "3-layer Edge-Gated MPNN", "Trained for Reaction-Center & Candidate Scoring"),
        ("Classical Baseline Ranker (GBDT)", "checkpoints/baselines/candidate_ranker.joblib", "100 Decision Trees, 1024-bit Morgan Fingerprints", "Trained on ECFP4 difference features"),
        ("RL Candidate Selector Policy", "checkpoints/rl/best_policy.pt", "Actor-Critic Policy Network", "Trained via Supervised Warm-Start + Policy Gradient"),
        ("Hybrid Calibration & Fusion Weights", "checkpoints/hybrid_fusion.json", "Logistic Platt Scaling", "Trained on Validation Hold-out"),
    ]

    models_info = []
    all_ready = True
    for name, p_str, arch, status in ckpts:
        p = Path(p_str)
        exists = p.exists()
        size_mb = round(p.stat().st_size / (1024 * 1024), 2) if exists else None
        if not exists:
            all_ready = False
        models_info.append(
            ModelCheckpointInfo(
                name=name,
                checkpoint_path=p_str,
                exists=exists,
                size_mb=size_mb,
                architecture=arch,
                trained_status=status if exists else "Missing Checkpoint",
            )
        )

    return ModelStatusResponse(
        status="ready" if all_ready else "degraded",
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        models=models_info,
        ready_for_inference=all_ready,
    )


@router.post("/reactions/predict", response_model=PredictionResponse)
def predict_reaction_standardized(req: PredictionRequest):
    """Standardized endpoint alias for reaction prediction."""
    return predict_reaction(req)


@router.post("/molecules/structure")
def generate_molecule_structure(req: StructureRequest):
    """
    Validates a SMILES string, generates 3D coordinates via ETKDGv3,
    optimizes with MMFF94 force field, and returns MOL block, SDF, and atom coordinates.
    """
    from chemrxn.chemistry.conformer import generate_3d_conformer
    res = generate_3d_conformer(
        smiles=req.smiles,
        add_hydrogens=req.add_hydrogens,
        optimize_mmff=req.optimize_mmff,
    )
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res.get("error", "Conformer generation failed"))
    return res


@router.post("/reactions/compare")
def compare_reactions_endpoint(req: ReactionCompareRequest):
    """
    Compares reactants and proposed product structure.
    Determines reacting centers, conserved substructures (MCS), and mass balance deltas.
    """
    from chemrxn.chemistry.conformer import compare_reaction_structures
    res = compare_reaction_structures(
        reactant_smiles=req.reactant_smiles,
        product_smiles=req.product_smiles,
    )
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res.get("error", "Reaction comparison failed"))
    return res


@router.get("/benchmarks")
def get_benchmarks_endpoint():
    """Returns stored benchmark evaluation results with full provenance and 95% bootstrap CIs."""
    return get_rigorous_evaluation_report()


@router.get("/experiments")
def get_experiments_endpoint():
    """Returns available training and evaluation experiment records."""
    import json
    manifest_path = Path("data/processed/split_manifest.json")
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return {
        "dataset_split_manifest": manifest,
        "active_models": [
            {"model": "Seq2Seq Transformer", "checkpoint": "checkpoints/transformer/best_checkpoint.pt"},
            {"model": "Molecular Reaction GNN", "checkpoint": "checkpoints/gnn/best_gnn.pt"},
            {"model": "Classical GBDT Ranker", "checkpoint": "checkpoints/baselines/candidate_ranker.joblib"},
            {"model": "RL Candidate Selector Policy", "checkpoint": "checkpoints/rl/best_policy.pt"},
        ],
        "evaluations_available": [
            "reports/evaluation/evaluation_results.json",
            "reports/rl_candidate_policy_report.json",
            "reports/hybrid_inference_ablation_report.json",
            "reports/gnn_candidate_ranking_report.json",
        ],
    }





