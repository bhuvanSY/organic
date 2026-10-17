# ChemRxn-Intelligence

> Organic Reaction Prediction Research Platform using Reactant SMILES as input.

Built from first principles for chemical synthesis research, ChemRxn-Intelligence translates reactant SMILES into predicted product SMILES and molecular fingerprints using custom neural architectures with randomly initialized weights (strictly zero pretrained chemical models or hosted APIs).

---

## Architectural Principles

1. **Modular Scientific Hierarchy**:
   - `chemrxn.chemistry`: Chemical parsing, atom-mapping stripping, RDKit canonicalization, molecular descriptors (MW, LogP, TPSA, Lipinski), and Morgan fingerprints (ECFP4/ECFP6).
   - `chemrxn.data`: Streaming and chunked loaders for 900,000+ patent reaction datasets (`US_patents_1976-Sep2016` & `Jin_USPTO`), regex SMILES tokenizers, PyTorch `Dataset` and Windows-safe `DataLoader` pipelines.
   - `chemrxn.models`: Abstract `BaseReactionModel` with strictly randomized scratch initialization, parameter accounting, `FingerprintReactionMLP`, and `Seq2SeqReactionTransformer`.
   - `chemrxn.training`: Mixed-precision (`torch.amp.autocast`), gradient accumulation, reproducible seeding, and masked cross-entropy / soft-Tanimoto losses.
   - `chemrxn.evaluation`: Empirical Top-k exact-match accuracy, chemical validity rates via RDKit, and Tanimoto similarity distributions.
   - `chemrxn.inference`: Reactant preprocessing, candidate generation, and RDKit product verification.
   - `chemrxn.api`: High-performance FastAPI server with Pydantic v2 schemas for SMILES validation, hardware telemetry, dataset metrics, and predictions.
   - `frontend`: React 18, TypeScript, and Vite research UI styled in warm ivory, sage green, muted olive, charcoal typography, and subtle grey borders.

2. **Laptop Optimization (NVIDIA RTX 3050 / RTX 4050)**:
   - Automated CUDA acceleration detection with graceful CPU fallback.
   - Mixed-precision (`fp16`) to fit sequence models within 4GB–6GB VRAM.
   - Gradient accumulation (`steps=4`) enabling effective batch sizes of 128+ without VRAM spikes.
   - Windows-safe dataloaders (`num_workers=0` by default to avoid process spawning overhead and GPU memory duplication).
   - Chunked and sample-limited streaming for 900,000-reaction datasets.

3. **Scientific Integrity**:
   - Random weight initialization from scratch.
   - Zero fabricated benchmark scores; untried or untrained models explicitly report status.

---

## Directory Structure

```text
chemrxn-intelligence/
├── chemrxn/                     # Core Python research package
│   ├── api/                     # FastAPI application, Pydantic v2 schemas, endpoints
│   ├── chemistry/               # RDKit validation, canonicalization, descriptors, Morgan FPs
│   ├── config/                  # AppConfig, Pydantic settings, YAML loader
│   ├── data/                    # Tokenizers, datasets, streaming loaders, collators
│   ├── evaluation/              # Top-k match, chemical validity, Tanimoto distributions
│   ├── inference/               # Inference engine and candidate verification
│   ├── models/                  # BaseReactionModel, FingerprintMLP, Seq2SeqTransformer
│   ├── training/                # ReactionTrainer, AMP, gradient accumulation, losses
│   └── utils/                   # CUDA/CPU device resolution, seed_everything, logger
├── configs/                     # System and training configurations
│   ├── default_config.yaml      # Standard research setup
│   └── rtx_laptop_config.yaml   # Tuned for RTX 3050/4050 4GB-6GB VRAM
├── data/                        # Processed caches and token vocabularies
├── frontend/                    # React + TypeScript + Vite research dashboard
├── ReactionSeq2Seq_Dataset/     # 900k+ reaction dataset files (Lowe US Patents & Jin USPTO)
├── tests/                       # Automated pytest suite (21 unit & integration tests)
├── pyproject.toml               # Package metadata and build configuration
├── requirements.txt             # Dependency specification
└── verify_setup.py              # Self-contained environment and foundation diagnostics
```

---

## Environment Setup & Commands

### 1. Verification Script
To perform a complete diagnostic check across Python, PyTorch CUDA, RDKit, seed reproducibility, scratch models, and datasets:
```powershell
python verify_setup.py
```

### 2. Running Automated Tests
Run the entire automated test suite:
```powershell
python -m pytest -v
```

### 3. Launching the Prediction API
```powershell
uvicorn chemrxn.api.app:app --host 127.0.0.1 --port 8001 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://127.0.0.1:8001/docs`.

### 4. Running the Frontend
```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.
