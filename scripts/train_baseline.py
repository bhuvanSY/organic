"""
Standalone training script for Classical ML Baselines.
Trains and persists:
1. Reaction-Class Classifier (XGBoost / GBDT)
2. Nearest-Neighbor Reaction Retrieval index
3. Supervised Candidate Ranker
"""

import sys
from chemrxn.baselines.runner import main

if __name__ == "__main__":
    main()
