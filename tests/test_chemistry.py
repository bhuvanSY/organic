"""
Unit tests for chemistry utilities, SMILES parsing, and fingerprints.
"""

import pytest
from chemrxn.chemistry.fingerprints import (
    compute_maccs_keys,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
)
from chemrxn.chemistry.reactions import clean_reaction_smiles, parse_reaction_smiles
from chemrxn.chemistry.smiles import (
    canonicalize_smiles,
    compute_molecular_descriptors,
    is_valid_smiles,
    strip_atom_mapping,
)


def test_is_valid_smiles():
    assert is_valid_smiles("CCO")
    assert is_valid_smiles("c1ccccc1")
    assert is_valid_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert not is_valid_smiles("InvalidSmilesString!!!")
    assert not is_valid_smiles("")


def test_strip_atom_mapping():
    mapped = "[CH3:1][C:2](=[O:3])[OH:4]"
    stripped = strip_atom_mapping(mapped)
    assert stripped == "CC(=O)O"
    assert ":" not in stripped


def test_canonicalize_smiles():
    s1 = "OCC"
    s2 = "CCO"
    assert canonicalize_smiles(s1) == canonicalize_smiles(s2) == "CCO"
    assert canonicalize_smiles("invalid") is None


def test_compute_molecular_descriptors():
    desc = compute_molecular_descriptors("CC(=O)O")
    assert desc is not None
    assert desc["molecular_formula"] == "C2H4O2"
    assert desc["molecular_weight"] == pytest.approx(60.05, abs=0.1)
    assert desc["num_heavy_atoms"] == 4


def test_morgan_fingerprints_and_tanimoto():
    fp_ethanol = compute_morgan_fingerprint("CCO", radius=2, nbits=2048)
    assert fp_ethanol.shape == (2048,)
    assert fp_ethanol.sum() > 0

    # Self-similarity must be exactly 1.0
    sim_self = compute_tanimoto_similarity("CCO", "CCO")
    assert sim_self == pytest.approx(1.0, abs=1e-5)

    # Ethanol vs Benzene should have low similarity
    sim_diff = compute_tanimoto_similarity("CCO", "c1ccccc1")
    assert 0.0 <= sim_diff < 0.3


def test_parse_and_clean_reaction():
    rxn = "[CH3:1][C:2](=[O:3])[Cl:4].[NH3:5]>>[CH3:1][C:2](=[O:3])[NH2:5]"
    parsed = parse_reaction_smiles(rxn)
    assert len(parsed["reactants"]) == 2
    assert len(parsed["products"]) == 1

    cleaned = clean_reaction_smiles(rxn)
    assert cleaned is not None
    assert ">" in cleaned
    assert ":" not in cleaned
