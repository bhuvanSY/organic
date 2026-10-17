"""
Reaction Feature Engineering using RDKit.
Extracts Morgan fingerprints, physicochemical descriptors, and reaction difference vectors.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors


class ReactionFeaturizer:
    """
    Constructs fixed-length feature vectors for organic reactions from reactants,
    reagents, and product candidates using RDKit.
    """

    def __init__(
        self,
        fp_radius: int = 2,
        fp_nbits: int = 1024,
        reagent_nbits: int = 256,
        include_descriptors: bool = True,
    ):
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.reagent_nbits = reagent_nbits
        self.include_descriptors = include_descriptors
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_nbits)
        self._reagent_gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=reagent_nbits)

    def mol_fingerprint(self, smiles: str, nbits: Optional[int] = None) -> np.ndarray:
        """Computes Morgan fingerprint as float32 numpy array."""
        target_bits = nbits or self.fp_nbits
        arr = np.zeros((target_bits,), dtype=np.float32)
        if not smiles:
            return arr
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                gen = self._gen if target_bits == self.fp_nbits else rdFingerprintGenerator.GetMorganGenerator(radius=self.fp_radius, fpSize=target_bits)
                fp = gen.GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(fp, arr)
        except Exception:
            pass
        return arr

    def compute_descriptors(self, smiles: str) -> np.ndarray:
        """
        Computes standard normalized physicochemical descriptors:
        [MW/500, LogP/5, TPSA/200, HBD/10, HBA/10, RotBonds/10, AromaticRings/5, Fsp3]
        """
        feats = np.zeros(8, dtype=np.float32)
        if not smiles:
            return feats
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.ExactMolWt(mol) / 500.0
                logp = Descriptors.MolLogP(mol) / 5.0
                tpsa = Descriptors.TPSA(mol) / 200.0
                hbd = Lipinski.NumHDonors(mol) / 10.0
                hba = Lipinski.NumHAcceptors(mol) / 10.0
                rot = Descriptors.NumRotatableBonds(mol) / 10.0
                arom = Lipinski.NumAromaticRings(mol) / 5.0
                fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
                feats = np.array([mw, logp, tpsa, hbd, hba, rot, arom, fsp3], dtype=np.float32)
        except Exception:
            pass
        return feats

    def featurize_reactants(
        self,
        reactants_smiles: str,
        reagents_smiles: str = "",
    ) -> np.ndarray:
        """
        Feature vector for reaction classification and retrieval:
        Concat [FP_reactants (1024), Descriptors_reactants (8), FP_reagents (256)]
        """
        r_fp = self.mol_fingerprint(reactants_smiles, nbits=self.fp_nbits)
        rg_fp = self.mol_fingerprint(reagents_smiles, nbits=self.reagent_nbits)

        if self.include_descriptors:
            r_desc = self.compute_descriptors(reactants_smiles)
            return np.concatenate([r_fp, r_desc, rg_fp])
        return np.concatenate([r_fp, rg_fp])

    def featurize_candidate_pair(
        self,
        reactants_smiles: str,
        candidate_smiles: str,
    ) -> np.ndarray:
        """
        Feature vector for supervised candidate ranking:
        [FP_reactants, FP_candidate, FP_diff, Tanimoto, delta_MW, delta_LogP, delta_TPSA]
        """
        r_fp = self.mol_fingerprint(reactants_smiles, nbits=self.fp_nbits)
        c_fp = self.mol_fingerprint(candidate_smiles, nbits=self.fp_nbits)
        diff_fp = c_fp - r_fp

        # Tanimoto similarity
        dot = np.dot(r_fp, c_fp)
        denom = np.dot(r_fp, r_fp) + np.dot(c_fp, c_fp) - dot
        tanimoto = float(dot / denom) if denom > 0 else 0.0

        r_desc = self.compute_descriptors(reactants_smiles)
        c_desc = self.compute_descriptors(candidate_smiles)
        diff_desc = c_desc - r_desc

        return np.concatenate([
            r_fp,
            c_fp,
            diff_fp,
            np.array([tanimoto], dtype=np.float32),
            diff_desc,
        ])
