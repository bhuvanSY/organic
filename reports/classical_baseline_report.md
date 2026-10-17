# ChemRxn-Intelligence: Classical ML Baseline Evaluation Report

> **Empirical Baseline Analysis without Pretrained Weights**

- **Execution Timestamp**: 2026-10-17T12:33:23.185389
- **Seed Determinism**: 42
- **Dataset Source**: data\processed\train.parquet

## 1. Critical Architectural Distinction: Classification vs. Generation

> [!IMPORTANT]
> **Reaction-Class Prediction vs. Product Prediction**:
> - **Reaction Classification** predicts which of the 10 transformation types (e.g., Acylation, Reduction, C-C Coupling) applies to the input reactants. It **never generates or synthesizes product SMILES**.
> - **Nearest-Neighbor Retrieval & Candidate Ranking** serve as the product-level baseline by searching historical patent reactions or re-ranking proposed product molecules.
> - **Generative Product Prediction** (custom Seq2Seq Transformer & Graph Neural Network) will be evaluated on exact-match SMILES sequence generation against these baselines.

## 2. Reaction-Class Classifier Results (XGBoost / GBDT)

- **Test Samples Evaluated**: 1,000
- **Overall Accuracy**: **73.80%**
- **Macro-F1 Score**: **0.4028**
- **Weighted-F1 Score**: **0.6839**
- **Inference Latency**: **0.063 ms/sample**

### Per-Class Performance Breakdown

| Reaction Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Heteroatom_Alkylation_Arylation | 0.745 | 0.965 | 0.841 | 627 |
| Acylation_and_Related | 0.779 | 0.669 | 0.720 | 142 |
| C_C_Bond_Formation | 0.000 | 0.000 | 0.000 | 9 |
| Heterocycle_Formation | 0.786 | 0.092 | 0.164 | 120 |
| Protections | 0.556 | 0.500 | 0.526 | 20 |
| Deprotections | 0.400 | 0.200 | 0.267 | 40 |
| Reductions | 0.583 | 0.206 | 0.304 | 34 |
| Oxidations | 1.000 | 0.250 | 0.400 | 8 |

## 3. Nearest-Neighbor Reaction Retrieval Baseline

- **Database Historical Reactions**: 5,000
- **Test Query Samples**: 1,000
- **Top-1 Exact Product Hit Rate**: **0.00%**
- **Top-3 Exact Product Hit Rate**: **0.00%**
- **Top-5 Exact Product Hit Rate**: **0.00%**
- **Mean Top-1 Tanimoto Similarity**: **0.2879**
- **Retrieval Latency**: **2.764 ms/query**

## 4. Supervised Candidate-Ranking Baseline

- **Cases Evaluated**: 500
- **Top-1 Candidate Selection Accuracy**: **94.80%**
- **Top-3 Candidate Selection Accuracy**: **98.80%**
- **Mean Reciprocal Rank (MRR)**: **0.9693**
- **Ranking Latency**: **18.709 ms/case**

## 5. Comparative Framework: Baseline vs. Custom Neural Architectures

| Model Family | Paradigm | Input Representation | Output Target | Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reaction Classifier** | Classical ML (XGBoost) | Morgan FP (1024) + Descriptors | Categorical Label (10 classes) | High throughput, interpretable | Cannot generate novel molecules |
| **Nearest-Neighbor** | Instance Retrieval | Cosine on ECFP4 vectors | Retrieved Historical Product | Non-parametric, zero training | Poor generalization on novel scaffolds |
| **Candidate Ranker** | Discriminative GBDT | Pair Difference Fingerprint | Plausibility Probability | Filters spurious proposals | Relies on external candidate generator |
| **Custom Transformer** | End-to-End Generative | Tokenized Reactant SMILES | Auto-regressive Product SMILES | Novel molecule generation, atom-rearrangement | Requires significant training epochs |
| **Graph Neural Model** | Geometric Message Passing | Molecular Atom-Bond Graph | Atom-center reactivity edits | Chemically bounded, physically rigorous | Graph parsing overhead |