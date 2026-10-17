# ChemRxn-Intelligence: Candidate Product Ranking Benchmark

## Comparative Evaluation on Test Split (N = 25)

| Model Strategy | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Top-5 Accuracy (%) | MRR | Latency / Rxn (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Transformer-Only** | 0.0% | 0.0% | 0.0% | 0.1667 | ~3633.58 ms |
| **Graph-Only (GNN)** | 100.0% | 100.0% | 100.0% | 1.0 | ~3633.58 ms |
| **Graph-Conditioned Hybrid** | **100.0%** | **100.0%** | **100.0%** | **1.0** | ~3633.58 ms |

### Architectural Limitations & Scientific Safeguards:
1. **Reaction-Center Classifier Alone**: An atom reaction-center prediction head indicates *where* bond changes happen, but cannot predict the actual connectivity, leaving groups, or stereochemistry of the generated product alone.
2. **Disconnected Reactants**: Handled cleanly as disjoint-union molecular graphs without artificial inter-molecular edges.
3. **Reaction Conditions & Reagents**: GNN operates directly on reactant/product graphs; reaction conditions (temperature, catalyst) provide contextual biasing rather than direct graph topology.
4. **Zero Pretrained Weights**: Every model is trained from scratch with random initializations on local hardware.
