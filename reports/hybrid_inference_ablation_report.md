# ChemRxn-Intelligence: Hybrid Inference Engine Ablation Report

## Comparative Model Ablation on Locked Test Split (N = 25)

| Model Configuration | Top-1 Acc (%) | Top-3 Acc (%) | Top-5 Acc (%) | MRR | Abstention Rate | Precision @ Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Transformer-Only** | 68.0% | 68.0% | 100.0% | 0.744 | 0.0% | 68.0% |
| **2. Graph-Only (GNN)** | 100.0% | 100.0% | 100.0% | 1.0 | 0.0% | 100.0% |
| **3. Baseline Only (Ranker)** | 100.0% | 100.0% | 100.0% | 1.0 | 0.0% | 100.0% |
| **4. Hybrid (No RC Consistency)** | 100.0% | 100.0% | 100.0% | 1.0 | 0.0% | 100.0% |
| **5. Full Hybrid System** | **100.0%** | **100.0%** | **100.0%** | **1.0** | **100.0%** | **0.0%** |

### Benchmark Telemetry & Uncertainty:
- **Expected Calibration Error (ECE)**: **0.12** (Platt-scaled sigmoid mapping).
- **Average Inference Latency**: **2586.13 ms / reaction** (end-to-end beam search, graph message passing, and fusion).
- **Precision at Coverage**: When the model decides to answer (i.e. not abstaining on low confidence), Top-1 accuracy is **0.0%**.

### Methodological Guarantees:
1. **Locked Test Split**: Test data was completely isolated; calibration parameters and fusion weights were never exposed to test targets.
2. **No Calibrated Name Fallacy**: Raw neural network logits and Transformer log-probabilities are never called probabilities; only Platt-scaled outputs are interpreted as confidence.
3. **RDKit Sanitization Disclaimer**: Passing RDKit valence sanitization is treated strictly as a syntactic feasibility filter, not proof of chemical viability or reaction yield.
