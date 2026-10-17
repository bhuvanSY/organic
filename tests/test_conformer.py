"""
Unit tests for 3D conformer generation and structure comparison.
"""

import pytest
from chemrxn.chemistry.conformer import (
    generate_3d_conformer,
    compare_reaction_structures,
)


def test_generate_3d_conformer_simple_benzene():
    res = generate_3d_conformer("c1ccccc1", optimize_mmff=True)
    assert res["success"] is True
    assert res["num_heavy_atoms"] == 6
    assert res["mol_block"] is not None
    assert "M  END" in res["mol_block"]
    assert res["sdf"] is not None
    assert "$$$$" in res["sdf"]
    assert res["energy_kcal_mol"] is not None
    assert len(res["atoms"]) > 0
    # Coordinates should be 3D
    for atom in res["atoms"]:
        assert "x" in atom and "y" in atom and "z" in atom


def test_generate_3d_conformer_stereochemistry():
    # Chiral molecule: L-alanine
    smiles_chiral = "C[C@@H](N)C(=O)O"
    res = generate_3d_conformer(smiles_chiral)
    assert res["success"] is True
    assert res["canonical_smiles"] is not None
    assert "N" in res["mol_block"]


def test_generate_3d_conformer_multi_fragment():
    # Disjoint union of two salts / reactants
    smiles_multi = "CC(=O)O.[Na+]"
    res = generate_3d_conformer(smiles_multi)
    assert res["success"] is True
    assert res["num_atoms"] > 0


def test_generate_3d_conformer_invalid_smiles():
    res = generate_3d_conformer("Not_A_Chemical_Formula_12345")
    assert res["success"] is False
    assert res["error"] is not None
    assert res["mol_block"] is None


def test_compare_reaction_structures():
    reactant = "CC(=O)Cl.c1ccccc1"
    product = "CC(=O)c1ccccc1"
    res = compare_reaction_structures(reactant, product)
    assert res["success"] is True
    assert res["num_conserved_atoms"] > 0
    assert res["product_reaction_center_atom_indices"] is not None
    assert "disclaimer" in res
