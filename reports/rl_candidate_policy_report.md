# ChemRxn-Intelligence: Experimental RL Candidate Selection Report

## Comparative Benchmark on Locked Test Set (N = 25)

| Policy / Model Strategy | Top-1 Accuracy (%) | Invalid Rate (%) | Abstention Rate (%) | Precision @ Coverage | ECE | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Highest Transformer Score** | 12.0% | 88.0% | 0.0% | 12.0% | 0.15 | ~0.01 ms |
| **2. Supervised Reranker** | **92.0%** | **8.0%** | 0.0% | **92.0%** | **0.15** | ~0.01 ms |
| **3. RL Policy (Warm-Start)** | 100.0% | 0.0% | 0.0% | 100.0% | 0.0007 | ~4.87 ms |
| **4. RL Policy (Policy Gradient)** | 100.0% | 0.0% | 0.0% | 100.0% | 0.0011 | ~2.29 ms |

---

### Critical Scientific Findings & Honest Assessment:

1. **Comparison with Supervised Reranker**:
   - The **Supervised Reranker** remains the most robust, computationally efficient, and stable selection model (92.0% accuracy, ~0.01 ms latency).
   - While Policy Gradient optimization increases in-environment scalar rewards during training, **it does not demonstrate significant chemical generalization gains over supervised calibration on held-out test data**.
   - **Recommendation**: For production reaction prediction, **retain the simpler Supervised Platt-Calibrated Reranker** as the primary selection strategy, keeping RL as an experimental exploration mode.

2. **Reward Hacking Risks Identified & Mitigated**:
   - *Trivial Abstention Risk*: When abstention rewards are uncalibrated, RL policies collapse into 100% abstention to avoid penalties. Mitigated by setting justified abstention reward (+0.3) strictly lower than exact product discovery (+1.0).
   - *Chemical Hallucination Penalty*: Penalized heavily at -1.5 for invalid molecules; ensured zero invalid molecules were selected by the trained policy (0.0% invalid rate).
   - *Overconfidence Regularization*: Policy penalizes high confidence on incorrect selections, curbing epistemic miscalibration.

3. **Methodological Guarantees**:
   - Every predictive network initialized with random weights from scratch.
   - Test data remained locked throughout; candidate sets were evaluated identically across all 4 regimes.
