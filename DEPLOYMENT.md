# ChemRxn-Intelligence — Production Deployment & Operation Guide

This guide details the dual-tier production deployment architecture for **ChemRxn-Intelligence**, a multi-modal organic chemical reaction prediction platform and interactive 3D molecular workbench.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │             Client Web Browser               │
                      └───────┬──────────────────────────────▲───────┘
                              │                              │
                    HTTPS     │                              │ WebGL Canvas
                   REST API   │                              │ (3Dmol.js)
                              ▼                              │
         ┌─────────────────────────────────────┐             │
         │          Vercel Frontend            │             │
         │  (React 19 + TypeScript + Vite)     │             │
         │  • Warm editorial light identity    │─────────────┘
         │  • 2D RDKit SVG Depiction           │
         │  • 3D Interactive Conformer Studio  │
         │  • Benchmark & Split Telemetry      │
         └──────────────────┬──────────────────┘
                            │
               HTTPS / CORS │ `VITE_API_BASE_URL`
                            ▼
         ┌─────────────────────────────────────────────────────┐
         │              Python Backend Service                 │
         │         (Container / Render / Fly.io / GPU)         │
         │  • FastAPI ASGI Engine (Uvicorn)                    │
         │  • RDKit 2026.09 (ETKDGv3 Conformer + MMFF94)       │
         │  • PyTorch 2.14 Neural Inference Engine             │
         │  • 5 Locally Trained Scratch Checkpoints:           │
         │    - Transformer Beam Generator (best_checkpoint.pt)│
         │    - Molecular Reaction GNN (best_gnn.pt)           │
         │    - Supervised GBDT Ranker (candidate_ranker)      │
         │    - RL Policy Selector (best_policy.pt)            │
         │    - Multi-Modal Fusion Calibrator (Platt Scaling)  │
         └─────────────────────────────────────────────────────┘
```

---

## 1. Frontend Deployment (Vercel)

The frontend is built with React 19, TypeScript, and Vite. It is pre-configured with `vercel.json` for seamless client-side SPA routing and long-term asset caching.

### Step 1: Connect Repository to Vercel
1. Log in to [Vercel Dashboard](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub / GitLab repository.

### Step 2: Configure Project Settings
- **Framework Preset**: `Vite`
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`

### Step 3: Environment Variables
Add the production backend URL under **Settings -> Environment Variables**:

| Variable Name | Value | Environment |
|---|---|---|
| `VITE_API_BASE_URL` | `https://chemrxn-backend.onrender.com` | Production & Preview |

*(Replace with your actual backend URL obtained in Section 2)*

---

## 2. Backend Deployment (Docker / Render / Fly.io)

> **Important Runtime Consideration**: RDKit C++ bindings and PyTorch require Linux system libraries (`libxrender1`, `libxext6`, `libgl1`) and sufficient memory (>= 2GB RAM). Therefore, the backend must run on a containerized service (Render, Fly.io, Railway, or AWS ECS/EC2) rather than standard Vercel serverless functions.

### Option A: Render (Zero-Config Blueprint)
The repository includes a ready-to-deploy `render.yaml` blueprint:
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **Blueprints** -> **New Blueprint Instance**.
3. Select this repository.
4. Render will parse `render.yaml` and configure:
   - Docker runtime using root `Dockerfile`
   - Health check path `/health`
   - Port `8001`
   - Configured CORS origins: `https://*.vercel.app`
5. Click **Apply**. Once built, copy your live backend URL (e.g. `https://chemrxn-api.onrender.com`).

### Option B: Fly.io CLI Deployment
1. Install `flyctl`:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
2. Launch the app:
   ```bash
   fly launch --name chemrxn-intelligence-api
   ```
3. Set environment variable:
   ```bash
   fly secrets set CORS_ORIGINS="https://*.vercel.app,http://localhost:5173"
   ```
4. Deploy:
   ```bash
   fly deploy
   ```

### Option C: Standalone Docker Run
To run the production backend anywhere (e.g., AWS EC2, GCP Compute Engine, local server):
```bash
# Build the production container
docker build -t chemrxn-backend:latest .

# Run with port 8001 mapped
docker run -d \
  -p 8001:8001 \
  -e PORT=8001 \
  -e CORS_ORIGINS="https://your-frontend.vercel.app,http://localhost:5173" \
  --name chemrxn-api \
  chemrxn-backend:latest
```

---

## 3. End-to-End Verification & Health Smoke Test

### 1. Backend Health Check
```bash
curl -s http://localhost:8001/health | jq .
# Expected output:
# {
#   "status": "ok",
#   "service": "ChemRxn-Intelligence API",
#   "version": "1.0.0"
# }
```

### 2. Model Checkpoint Registry Status
```bash
curl -s http://localhost:8001/api/v1/model/status | jq .
# Expected output:
# {
#   "status": "ready",
#   "ready_for_inference": true,
#   "models": [ ... 5 verified local checkpoints ... ]
# }
```

### 3. Real 3D Conformer Generation
```bash
curl -s -X POST http://localhost:8001/api/v1/molecules/structure \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1", "optimize_force_field": true}' | jq '{success, num_atoms, energy_kcal_mol, force_field}'
# Expected output:
# {
#   "success": true,
#   "num_atoms": 12,
#   "energy_kcal_mol": 10.377,
#   "force_field": "MMFF94"
# }
```

---

## 4. Security & Scientific Integrity Principles

1. **Authentication & Data Privacy**: All API communications use standard JSON serialization over HTTPS. No filesystem paths, internal traces, or arbitrary code execution are permitted.
2. **Deterministic Reproducibility**: Random seed `42` is locked across all model evaluations and scaffold splits.
3. **Chemical Ground Truth Disclaimer**: Computational 3D conformers generated by ETKDGv3 and MMFF94 energy minimizations are computational approximations, clearly labeled as non-crystal structures.
4. **Zero Model Hallucination**: No benchmark numbers, predictions, or synthetic metrics are fabricated. If any model is unavailable, the API returns a status code and degradation notice rather than simulated results.
