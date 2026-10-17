"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from chemrxn.api.app import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ChemRxn-Intelligence API"


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "torch_version" in data
    assert "rdkit_version" in data
    assert "cuda_available" in data


def test_api_hardware():
    response = client.get("/api/v1/hardware")
    assert response.status_code == 200
    data = response.json()
    assert "device_name" in data
    assert "device_type" in data


def test_api_validate_smiles_valid():
    response = client.post(
        "/api/v1/validate-smiles",
        json={"smiles": "[CH3:1][C:2](=[O:3])[OH:4]", "remove_mapping": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["canonical_smiles"] == "CC(=O)O"
    assert data["descriptors"]["molecular_formula"] == "C2H4O2"


def test_api_validate_smiles_invalid():
    response = client.post(
        "/api/v1/validate-smiles",
        json={"smiles": "not_a_molecule!!!", "remove_mapping": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert data["error"] is not None


def test_api_dataset_info():
    response = client.get("/api/v1/dataset-info")
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] >= 6
    assert data["total_size_mb"] > 900


def test_api_baseline_endpoints():
    # 1. Classify
    res_cls = client.post(
        "/api/v1/baseline/classify",
        json={"reactant_smiles": "CC(=O)Cl.CCN"},
    )
    assert res_cls.status_code == 200
    d_cls = res_cls.json()
    assert "predicted_class" in d_cls
    assert d_cls["is_generative"] is False

    # 2. Retrieve
    res_ret = client.post(
        "/api/v1/baseline/retrieve",
        json={"reactant_smiles": "CC(=O)Cl.CCN", "num_candidates": 3},
    )
    assert res_ret.status_code == 200
    d_ret = res_ret.json()
    assert len(d_ret["candidates"]) <= 3

    # 3. Rank Candidates
    res_rnk = client.post(
        "/api/v1/baseline/rank-candidates",
        json={
            "reactants": "CC(=O)Cl.CCN",
            "candidates": ["CC(=O)NCC", "c1ccccc1"],
        },
    )
    assert res_rnk.status_code == 200
    d_rnk = res_rnk.json()
    assert len(d_rnk["ranked_candidates"]) == 2
    assert d_rnk["ranked_candidates"][0]["candidate_smiles"] == "CC(=O)NCC"


def test_api_predict_endpoint():
    res = client.post(
        "/api/v1/predict",
        json={"reactant_smiles": "CC(=O)Cl.CCN", "num_candidates": 2, "beam_width": 2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["reactants_canonical"] in ["CCN.CC(=O)Cl", "CC(=O)Cl.CCN"]
    assert "candidates" in data


def test_api_reaction_centers_endpoint():
    res = client.post(
        "/api/v1/predict/reaction-centers",
        json={"reactant_smiles": "CC(=O)Cl.CCN"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["num_atoms"] == 7
    assert len(data["atom_reaction_probabilities"]) == 7


def test_api_benchmark_report_endpoint():
    res = client.get("/api/v1/reports/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "graph_conditioned_hybrid" in data
    assert "transformer_only" in data


def test_api_root_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_api_model_status():
    res = client.get("/api/v1/model/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "models" in data
    assert len(data["models"]) >= 4
    # All 5 locally trained checkpoints exist
    for m in data["models"]:
        assert m["exists"] is True


def test_api_render_svg():
    res = client.get("/api/v1/render-svg?smiles=CC(=O)c1ccccc1")
    assert res.status_code == 200
    assert "image/svg+xml" in res.headers["content-type"]
    assert "<svg" in res.text


def test_api_molecule_structure_3d_valid():
    res = client.post(
        "/api/v1/molecules/structure",
        json={"smiles": "c1ccccc1", "optimize_force_field": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["num_heavy_atoms"] == 6
    assert data["num_atoms"] == 12  # C6 + H6
    assert data["mol_block"] is not None
    assert len(data["atoms"]) == 12
    assert "disclaimer" in data


def test_api_molecule_structure_3d_invalid():
    res = client.post(
        "/api/v1/molecules/structure",
        json={"smiles": "INVALID_CHEMISTRY_XXXX"},
    )
    assert res.status_code == 400
    data = res.json()
    assert "detail" in data


def test_api_reaction_compare():
    res = client.post(
        "/api/v1/reactions/compare",
        json={"reactant_smiles": "CC(=O)Cl.c1ccccc1", "product_smiles": "CC(=O)c1ccccc1"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["num_conserved_atoms"] > 0
    assert data["product_reaction_center_atom_indices"] is not None


def test_api_benchmarks_and_experiments():
    res_b = client.get("/api/v1/benchmarks")
    assert res_b.status_code == 200
    b_data = res_b.json()
    assert "model_summaries" in b_data
    assert "validity_breakdown" in b_data

    res_e = client.get("/api/v1/experiments")
    assert res_e.status_code == 200
    e_data = res_e.json()
    assert "dataset_split_manifest" in e_data
    assert "active_models" in e_data






