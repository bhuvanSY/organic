"""
Automated unit tests for ChemRxn-Intelligence Classical ML Baselines.
Verifies featurization, reaction classifier, nearest-neighbor retriever, and candidate ranker.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from chemrxn.baselines.candidate_ranker import SupervisedCandidateRanker
from chemrxn.baselines.evaluator import BaselineEvaluator
from chemrxn.baselines.featurizer import ReactionFeaturizer
from chemrxn.baselines.nearest_neighbor import NearestNeighborRetriever
from chemrxn.baselines.reaction_classifier import ReactionClassClassifier
from chemrxn.baselines.taxonomy import CLASS_TO_ID, REACTION_CLASSES, ReactionClassifierTagger


def test_reaction_featurizer_dimensions():
    featurizer = ReactionFeaturizer(fp_nbits=512, reagent_nbits=128, include_descriptors=True)
    r_smi = "CC(=O)Cl.CCN"
    rg_smi = "c1ccncc1"

    feat = featurizer.featurize_reactants(r_smi, rg_smi)
    expected_dim = 512 + 8 + 128  # 648
    assert feat.shape == (expected_dim,)
    assert not np.isnan(feat).any()

    # Featurize candidate pair
    cand_smi = "CC(=O)NCC"
    pair_feat = featurizer.featurize_candidate_pair(r_smi, cand_smi)
    # pair feat: [r_fp(512), c_fp(512), diff_fp(512), tanimoto(1), diff_desc(8)] = 1545
    assert pair_feat.shape == (512 * 3 + 1 + 8,)
    assert not np.isnan(pair_feat).any()


def test_reaction_classifier_tagger():
    tagger = ReactionClassifierTagger()
    # Acylation
    c1 = tagger.classify_reaction("CC(=O)Cl.CCN", "CC(=O)NCC")
    assert c1 == "Acylation_and_Related"

    # Reduction
    c2 = tagger.classify_reaction("[O-][N+](=O)c1ccccc1", "Nc1ccccc1")
    assert c2 == "Reductions"


def test_reaction_class_classifier_train_and_predict():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create small synthetic feature dataset
        rng = np.random.RandomState(42)
        X = rng.randn(40, 64)
        y = rng.randint(0, 4, size=40)

        classifier = ReactionClassClassifier(model_type="hist_gb", seed=42, n_estimators=10)
        classifier.fit(X, y)
        assert classifier.is_trained

        preds = classifier.predict(X[:5])
        assert len(preds) == 5
        probs = classifier.predict_proba(X[:5])
        assert probs.shape[0] == 5
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Persistence test
        ckpt_path = Path(tmp_dir) / "test_cls.joblib"
        classifier.save(ckpt_path)
        assert ckpt_path.exists()

        loaded = ReactionClassClassifier.load(ckpt_path)
        loaded_preds = loaded.predict(X[:5])
        assert np.array_equal(preds, loaded_preds)


def test_nearest_neighbor_retriever():
    with tempfile.TemporaryDirectory() as tmp_dir:
        retriever = NearestNeighborRetriever(metric="cosine", fp_nbits=256)
        reactants = ["CC(=O)Cl.CCN", "CC(=O)O.CCO", "c1ccccc1.Cl", "c1ccncc1.Br"]
        products = ["CC(=O)NCC", "CCOC(C)=O", "Clc1ccccc1", "Brc1ccncc1"]
        retriever.fit(reactants, products)

        # Query identical reactants - should retrieve itself at rank 1
        results = retriever.retrieve("CC(=O)Cl.CCN", top_k=2)
        assert len(results) == 2
        assert results[0]["retrieved_reactants"] == "CC(=O)Cl.CCN"
        assert results[0]["retrieved_product"] == "CC(=O)NCC"
        assert results[0]["similarity_score"] == pytest.approx(1.0, abs=1e-3)

        # Persistence test
        ckpt_path = Path(tmp_dir) / "test_nn.joblib"
        retriever.save(ckpt_path)
        assert ckpt_path.exists()

        loaded = NearestNeighborRetriever.load(ckpt_path)
        res_loaded = loaded.retrieve("CC(=O)Cl.CCN", top_k=1)
        assert res_loaded[0]["retrieved_product"] == "CC(=O)NCC"


def test_candidate_ranker():
    with tempfile.TemporaryDirectory() as tmp_dir:
        featurizer = ReactionFeaturizer(fp_nbits=256, include_descriptors=True)
        r = "CC(=O)Cl.CCN"
        true_p = "CC(=O)NCC"
        decoy1 = "c1ccccc1"
        decoy2 = "CCCC(=O)O"

        # Positive pair
        pos_feat = featurizer.featurize_candidate_pair(r, true_p)
        # Negative pairs
        neg_feat1 = featurizer.featurize_candidate_pair(r, decoy1)
        neg_feat2 = featurizer.featurize_candidate_pair(r, decoy2)

        X = np.stack([pos_feat, neg_feat1, neg_feat2] * 10, axis=0)
        y = np.array([1, 0, 0] * 10)

        ranker = SupervisedCandidateRanker(model_type="hist_gb", seed=42, fp_nbits=256)
        ranker.fit(X, y)

        scored = ranker.score_candidates(r, [decoy1, true_p, decoy2])
        # True product should be ranked #1
        assert scored[0][0] == true_p
        assert scored[0][1] > scored[1][1]

        # Persistence test
        ckpt_path = Path(tmp_dir) / "test_ranker.joblib"
        ranker.save(ckpt_path)
        assert ckpt_path.exists()

        loaded = SupervisedCandidateRanker.load(ckpt_path)
        scored_loaded = loaded.score_candidates(r, [decoy1, true_p])
        assert scored_loaded[0][0] == true_p


def test_baseline_evaluator():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    metrics = BaselineEvaluator.evaluate_classifier(y_true, y_pred, latency_ms_per_sample=1.5)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == pytest.approx(5 / 6, abs=0.01)
    assert metrics["latency_ms_per_sample"] == 1.5
