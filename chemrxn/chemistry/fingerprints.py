"""
Molecular fingerprint computation and similarity metrics.
"""

from typing import List, Optional, Union
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    nbits: int = 2048,
    as_numpy: bool = True,
) -> Optional[Union[np.ndarray, DataStructs.ExplicitBitVect]]:
    """
    Computes Morgan fingerprint (ECFP) for a given SMILES string.
    Radius 2 corresponds to ECFP4; radius 3 corresponds to ECFP6.

    Returns:
        numpy float32 array of shape (nbits,) or RDKit ExplicitBitVect, or None if invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        fp = gen.GetFingerprint(mol)
        
        if as_numpy:
            arr = np.zeros((nbits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        return fp
    except Exception:
        return None


def compute_maccs_keys(
    smiles: str,
    as_numpy: bool = True,
) -> Optional[Union[np.ndarray, DataStructs.ExplicitBitVect]]:
    """
    Computes 166-bit MACCS structural keys for a given SMILES string.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        if as_numpy:
            arr = np.zeros((167,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            # Typically bit 0 is not used in MACCS; return 166 or full 167
            return arr
        return fp
    except Exception:
        return None


def compute_tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2, nbits: int = 2048) -> float:
    """
    Computes Tanimoto similarity between the Morgan fingerprints of two SMILES.
    Returns a score in [0.0, 1.0]. Returns 0.0 if either SMILES is invalid.
    """
    fp1 = compute_morgan_fingerprint(smiles1, radius=radius, nbits=nbits, as_numpy=False)
    fp2 = compute_morgan_fingerprint(smiles2, radius=radius, nbits=nbits, as_numpy=False)
    
    if fp1 is None or fp2 is None:
        return 0.0
    
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def batch_tanimoto_similarity(preds: List[str], targets: List[str], radius: int = 2, nbits: int = 2048) -> List[float]:
    """
    Computes pairwise Tanimoto similarities between predicted and target SMILES lists.
    """
    return [
        compute_tanimoto_similarity(p, t, radius=radius, nbits=nbits)
        for p, t in zip(preds, targets)
    ]
