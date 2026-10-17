"""
SMILES chemical validation, canonicalization, atom-map stripping, and descriptors.
"""

from typing import Any, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def parse_mol_safely(smiles: str) -> Optional[Chem.Mol]:
    """
    Safely parses a SMILES string into an RDKit Mol object.
    Returns None if parsing fails.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        return mol
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """
    Checks if a SMILES string can be parsed and sanitized by RDKit.
    """
    mol = parse_mol_safely(smiles)
    return mol is not None


def strip_atom_mapping(smiles: str) -> str:
    """
    Removes atom-mapping numbers (e.g., :1, :14) from SMILES while preserving
    stereochemistry, aromaticity, and chemical validity.
    """
    mol = parse_mol_safely(smiles)
    if mol is None:
        # Fallback to regex if RDKit cannot parse mapped fragment
        import re
        return re.sub(r":\d+", "", smiles)
    
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            atom.SetAtomMapNum(0)
    
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_smiles(smiles: str, remove_mapping: bool = True) -> Optional[str]:
    """
    Standardizes a SMILES string into its canonical representation.
    Returns None if the SMILES is chemically invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    s = smiles.strip()
    if not s:
        return None
    
    if remove_mapping:
        mol = parse_mol_safely(s)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        mol = parse_mol_safely(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)


def compute_molecular_descriptors(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Computes key physicochemical descriptors for a given SMILES string.
    Returns dictionary of descriptors or None if invalid.
    """
    mol = parse_mol_safely(smiles)
    if mol is None:
        return None
    
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.ExactMolWt(mol), 3),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_h_donors": Lipinski.NumHDonors(mol),
        "num_h_acceptors": Lipinski.NumHAcceptors(mol),
        "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
        "num_saturated_rings": Lipinski.NumSaturatedRings(mol),
    }
